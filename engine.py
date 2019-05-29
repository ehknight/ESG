# NOTE: REVIEW ESG FILE; NEW DETAILS AND STUFF HAS CHANGED
# NOTE: BIDS ARE PLACED ON A DAY-TO-DAY BASIS
# also make sure to take integral in the case of sloped demand curve


import numpy as np
import scipy.optimize
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from time import sleep
from warnings import warn
from utils import reset_bids

# Globals

TESTING = False

if TESTING:
    demand_csv = pd.read_csv("test_csv/demand.csv")   
    portfolio_csv = pd.read_csv("test_csv/portfolios.csv")
    users_csv = pd.read_csv("test_csv/users.csv")
else:
    demand_csv = pd.read_csv("csv/demand.csv")
    portfolio_csv = pd.read_csv("csv/portfolios.csv")
    print(portfolio_csv)
    users_csv = pd.read_csv("csv/users.csv")

# Utility Functions
first = lambda x: x[0]
second = lambda x: x[1]
lmap = lambda f, x: list(map(f, x))

def print_players_from_plants(plants):
    for player in set([plant.owner for plant in plants]):
        print(player)

def create_step_function(steps):
    xs = list(map(first, steps))
    ys = list(map(second, steps))
    # get_break_point is the first point at which load <= x

    def get_break_point(load):
        try:
            sum_xs = [sum(xs[:i+1]) for i, _ in enumerate(xs)]
            assert len(xs) == len(sum_xs)
            x_greater_than_load = lmap(lambda x: load <= x, sum_xs)
            x = x_greater_than_load.index(True)
        except ValueError:
            warn("Couldn't find a valid breakpoint")
            x = 0
        print('load is at {}'.format(load))
        return x
    return lambda load: ys[get_break_point(load)], lambda load: get_break_point(load)

def get_intersection(f1, f2):
    if isinstance(f2, int) or isinstance(f2, float):
        return f2
    else:
        optim = scipy.optimize.minimize(lambda x: (f2(x[0]) - f1(x[0]))**2, x0=0)
        print("x1: {}, y1: {}, y2: {}".format(optim.x[0], f1(optim.x), f2(optim.x)))
        return optim.x[0]

# Game Objects
class GameState(object):
    def __init__(self):
        self.noise_scale = 0
        self.auction_type = 'uniform' # can also be discrete
        self.cur_day = 1
        self.cur_hour = 1
        self.breakpoints = []
        self.demands = []
        self.str_sorted_bids = []

        self.get_cur_demand_row = lambda: demand_csv[demand_csv['round'] == self.cur_day]\
                                         [demand_csv['hour'] == self.cur_hour]
        self.gen_noise = partial(np.random.normal, scale=self.noise_scale)
        def demand_fn(load, base_demand):
            cur_loadslope = sum(self.get_cur_demand_row()['loadslope'])
            assert cur_loadslope != 0
            return cur_loadslope * (load-base_demand)
        self.demand_fn = demand_fn

    def construct_price_curve(self, plant_bids):
        # plant_bids is tuples of (plant, bid)
        sorted_plants = sorted(plant_bids, key=second)
        plants_capacity = [(plant.capacity, bid) for plant, bid in sorted_plants]
        step_fn, breakpoint_fn = create_step_function(plants_capacity)
        return step_fn, sorted_plants, breakpoint_fn

    def construct_demand_curve(self):
        est_demand = self.get_cur_demand_row()['load'].tolist()
        cur_loadslope = self.get_cur_demand_row()['loadslope'].tolist()

        assert (len(est_demand) == 1) and (len(cur_loadslope) == 1)
        # assert cur_loadslope[0] == cur_loadslope[1]
        
        cur_loadslope = cur_loadslope[0]
        est_demand = est_demand[0]
        base_demand = est_demand+self.gen_noise()

        if cur_loadslope == 0: # NOTE: when loadslope is 0, treat as vertical line
            return base_demand
        else: 
            return partial(self.demand_fn, base_demand=base_demand)
    
    def switch_auction_type(self):
        if self.cur_day == 1 or self.cur_day == 2:
            self.auction_type = 'uniform'
        else:
            self.auction_type = 'discrete'

    def run_hour(self, plant_bids, auto_end_day=False):
        # tuples of (plant, bid)
        plants, bids = list(map(first, plant_bids)), list(map(second, plant_bids))
        price_fn, sorted_plants, breakpoint_fn = self.construct_price_curve(plant_bids)

        self.str_sorted_bids = [(plant.name, bid) for plant, bid in sorted_plants]
        print(self.str_sorted_bids)

        demand_fn = self.construct_demand_curve()
        true_demand = get_intersection(price_fn, demand_fn)
        # self.breakpoints.append(breakpoint_fn(true_demand))
        total_activated_so_far = 0

        if self.auction_type == 'uniform':
            price_per_mwh_fn = lambda plant: price_fn(true_demand)
        elif self.auction_type == 'discrete':
            price_per_mwh_fn = lambda plant: bids[plants.index(plant)]
        else:
            raise ValueError

        for plant, bid in sorted_plants:
            leftover_electricity = true_demand - total_activated_so_far
            assert leftover_electricity >= 0
            
            if leftover_electricity == 0:
                print("All electricity used up")

            electricity_used = min(leftover_electricity, plant.capacity)
            price_per_mwh = price_per_mwh_fn(plant)
            print("Plant {} used {} MWh of electricity at ${}/MWh"\
                  .format(plant.name, electricity_used, price_per_mwh))

            plant.log_cost(electricity_used)
            plant.log_profit(price_per_mwh, electricity_used)
            
            total_activated_so_far += electricity_used
        
        print_players_from_plants(plants)
        reset_bids(plants)
        self.cur_hour += 1
        if self.cur_hour > 4:
            self.end_day(plants)
        self.demands.append(true_demand)

    def end_day(self, plants):
        self.demands = []
        self.breakpoints = []
        for plant in plants:
            plant.transfer(day_end=True)
            plant.reset()

        self.switch_auction_type()
        print("Day ended")
        print_players_from_plants(plants)
        self.cur_hour = 1
        self.cur_day += 1

class Player(object):
    def __init__(self, name, starting_money):
        self.name = name
        self.plants = []
        self.bids = {}
        self.money = float(starting_money)
    
    def buy_portfolio(self, portfolio_name):
        plants = []
        raw_plants = portfolio_csv[portfolio_csv['portfolioname'] == portfolio_name]
        assert len(raw_plants) > 0
        for raw_plant_tup in raw_plants.iterrows():
            raw_plant = raw_plant_tup[1] # access second element bc. first elem is row #
            kwargs = {'name': raw_plant['name'].capitalize(),
                      'capacity': raw_plant['mw'],
                      'cost_per_mwh': raw_plant['fuelcost'] + raw_plant['varom'],
                      'heat_rate': 0, # NOTE: portfolio doesn't define heat rate
                      'fuel_price': 0, # NOTE: portfolio doesn't define fuel price
                      'o_and_m': raw_plant['fixom']}
            plant = Plant(**kwargs)
            plant.owner = self
            self.plants.append(plant)
            self.bids[plant] = 0
        
    def bid(self, plant_name, bid):
        matching_plants = [plant for plant in self.plants if plant.name==plant_name]
        assert len(matching_plants) == 1
        plant = matching_plants[0]
       
        self.bids[plant] = bid
        plant.bid = bid
    
    def __str__(self):
        return "\nName: {} | Plants owned: {} | Total money: ${}"\
                .format(self.name, [plant.name for plant in self.plants], self.money)

class Plant(object):
    def __init__(self, name, capacity, cost_per_mwh, 
                 heat_rate, fuel_price, o_and_m):
        # mw -> capacity; fuelcost+varom -> cost; 
        self.name = name
        self.capacity = capacity # MW
        self.cost_per_mwh = cost_per_mwh  # total var cost $/MWh
        self.heat_rate = heat_rate # MMBTU/MWh
        self.fuel_price = fuel_price # $/MMBTU
        self.o_and_m = o_and_m # $/day
        self.owner = None
        self.reset()
    
    def reset(self):
        # NOTE: when is o&m paid?
        self.costs = []
        self.profits = []
        self.electricity_used = []
        print("Resetting. Name: {}, Costs: {}, Profits: {}, Electricity Used: {}".format(self.name, self.costs, self.profits, self.electricity_used))

        self.bid = 0

    def log_cost(self, mwh_used):
        # calculate cost for some hour
        assert 0 <= mwh_used <= self.capacity
        heat_produced = mwh_used * self.heat_rate
        cost_from_heat = heat_produced * self.fuel_price
        cost_from_var_cost = mwh_used * self.cost_per_mwh
        cur_cost = cost_from_heat + cost_from_var_cost
        self.costs.append(cur_cost)
        self.electricity_used.append(mwh_used)

    def log_profit(self, price_per_mwh, amount_used):
        assert 0 <= amount_used <= self.capacity
        self.profits.append(price_per_mwh * amount_used)
    
    def transfer_cost(self, day_end=False):
        if day_end:
            self.costs.append(self.o_and_m)
        total_costs = sum(self.costs)
        self.owner.money -= total_costs
        self.costs = []
    
    def transfer_profit(self):
        total_profits = sum(self.profits)
        print("Transfering {:,} to {}".format(total_profits, self.owner.name))
        self.owner.money += total_profits
        self.profits = []
    
    def transfer(self, day_end=False):
        self.transfer_cost(day_end)
        self.transfer_profit()
