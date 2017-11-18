import numpy as np
import scipy.optimize
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from time import sleep

# Globals
STARTING_MONEY = 10000
demand_csv = pd.read_csv("/Users/ethknig/School 2017-2018/Environmental Econ/WorldNS/demand.csv")
portfolio_csv = pd.read_csv("/Users/ethknig/School 2017-2018/Environmental Econ/WorldNS/portfolios.csv")

# Utility Functions
first = lambda x: x[0]
second = lambda x: x[1]

def create_step_function(steps):
    xs = map(first, steps)
    ys = map(second, steps)
    # get_break_point is the first point at which x <= load

    def get_break_point(load):
        try:
            sum_xs = [sum(xs[:i+1]) for i, _ in enumerate(xs)]
            assert len(xs) == len(sum_xs)
            x = list(map(lambda x: x<=load, sum_xs)).index(1) + 1
        except ValueError:
            x = 0
        return x
    
    return lambda load: ys[get_break_point(load)]

def get_intersection(f1, f2):
    optim = scipy.optimize.minimize(lambda x: (f2(x[0]) - f1(x[0]))**2, x0=0, method='Nelder-Mead')
    print "x1: {}, y1: {}, y2: {}".format(optim.x[0], f1(optim.x), f2(optim.x))
    return optim.x[0]

# Game Objects
class GameState(object):
    def __init__(self):
        self.noise_scale = 100
        self.load_slope = -100
        self.demand_fn = lambda load, base_demand: -100 * (load-base_demand)

        self.auction_type = 'uniform' # can also be uniform
        self.cur_day = 1
        self.cur_hour = 1
        self.gen_noise = partial(np.random.normal, scale=self.noise_scale)

    def construct_price_curve(self, plant_bids):
        # tuples of (plant, bid)
        sorted_plants = sorted(plant_bids, key=second)
        plants_capacity = [(plant.capacity, bid) for plant, bid in sorted_plants]
        return create_step_function(plants_capacity), sorted_plants

    def construct_demand_curve(self):
        est_demand = demand_csv[demand_csv['round'] == self.cur_day] \
                               [demand_csv['hour'] == self.cur_hour]['load']
        assert len(est_demand) == 2
        est_demand = sum(est_demand) / float(len(est_demand))
        base_demand = est_demand+self.gen_noise()
        return partial(self.demand_fn, base_demand=base_demand)
    
    def switch_auction_type(self):
        if self.auction_type == 'uniform': 
            self.auction_type = 'discrete'
        elif self.auction_type == 'discrete':
            self.auction_type = 'uniform'
        else:
            raise ValueError

    def run_hour(self, plant_bids):
        # tuples of (plant, bid)
        plants, bids = map(first, plant_bids), map(second, plant_bids)
        price_fn, sorted_plants = self.construct_price_curve(plant_bids)

        print "Sorted bids: {}\n".format([(plant.name, bid)for plant, bid in sorted_plants])

        demand_fn = self.construct_demand_curve()
        true_demand = get_intersection(price_fn, demand_fn)
        total_activated_so_far = 0

        if self.auction_type == 'uniform':
            price_per_mwh_fn = lambda plant: price_fn(true_demand)
        elif self.auction_type == 'discrete':
            raise NotImplementedError
            price_per_mwh_fn = lambda plant: bids[plants.index(plant)]
        else:
            raise ValueError

        for plant, bid in sorted_plants:
            leftover_electricity = true_demand - total_activated_so_far
            electricity_used = min(leftover_electricity, plant.capacity)
            _, mwh_used = plant.activate(electricity_used)
            price_per_mwh = price_per_mwh_fn(mwh_used)
            print "Plant {} used {} MWh of electricity at ${}/MWh"\
                  .format(plant.name, electricity_used, price_per_mwh)
            plant.transfer_profits(price_per_mwh, mwh_used)
            total_activated_so_far -= electricity_used

    def end_day(self, plants):
        for plant in plants:
            total_cost = plant.reset()
            plant.owner.money -= total_cost
        self.switch_auction_type()

class Player(object):
    def __init__(self, name):
        self.name = name
        self.plants = []
        self.money = STARTING_MONEY
    
    def buy_portfolio(self, portfolio_name):
        plants = []
        raw_plants = portfolio_csv[portfolio_csv['portfolioname'] == portfolio_name]
        for raw_plant_tup in raw_plants.iterrows():
            raw_plant = raw_plant_tup[1] # access second element bc. first elem is row #
            kwargs = {'name': raw_plant['name'].capitalize(),
                      'capacity': raw_plant['mw'],
                      'cost': raw_plant['fuelcost'] + raw_plant['varom'],
                      'heat_rate': 0, # NOTE: portfolio doesn't define heat rate
                      'fuel_price': 0, # NOTE: portfolio doesn't define fuel price
                      'o_and_m': raw_plant['fixom']}
            plant = Plant(**kwargs)
            plant.owner = self
            self.plants.append(plant)
    
    def __str__(self):
        return "Player name: {} | Plants owned: {} | Total money: {} \n"\
                .format(self.name, [plant.name for plant in self.plants], self.money)

class Plant(object):
    def __init__(self, name, capacity, cost, 
                 heat_rate, fuel_price, o_and_m):
        # mw -> capacity; fuelcost+varom -> cost; 
        self.name = name
        self.capacity = capacity # MW
        self.cost = cost  # total var cost $/MWh
        self.heat_rate = heat_rate # MMBTU/MWh
        self.fuel_price = fuel_price # $/MMBTU
        self.o_and_m = o_and_m # $/day

        self.owner = None
        self.total_electricity = 0
        self.total_cost = 0
        self.reset()
    
    def reset(self):
        cost_this_day = self.total_cost + self.o_and_m
        self.total_electricity = 0
        self.total_cost = 0
        return cost_this_day

    def activate(self, mwh_used):
        # calculate cost for some hour
        assert 0 <= mwh_used <= self.capacity
        heat_produced = mwh_used * self.heat_rate
        cost_from_heat = heat_produced * self.fuel_price
        cost_from_var_cost = mwh_used * self.cost
        cur_cost = cost_from_heat + cost_from_var_cost
        self.total_cost += cur_cost
        self.total_electricity += mwh_used
        return cur_cost, mwh_used

    def transfer_profits(self, price_per_mwh, amount_used):
        assert 0 <= amount_used <= self.capacity
        print "Transfering {:,} to {} \n".format(price_per_mwh * amount_used, self.owner.name)
        self.owner.money += price_per_mwh * amount_used