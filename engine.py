# NOTE: REVIEW ESG FILE; NEW DETAILS AND STUFF HAS CHANGED
# NOTE: BIDS ARE PLACED ON A DAY-TO-DAY BASIS
# also make sure to take integral in the case of sloped demand curve

from collections import defaultdict
from collections.abc import Iterable

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN
from bokeh.embed import file_html, components

import numpy as np
import scipy.optimize
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from time import sleep
from warnings import warn
from utils import reset_bids

import csv

# Globals
interest_rate = 0.05 # %/day
carbon_tax_rate = 30 # $/ton

TESTING = False

if TESTING:
    demand_csv = pd.read_csv("test_csv/demand.csv")   
    portfolio_csv = pd.read_csv("test_csv/portfolios.csv")
    users_csv = pd.read_csv("test_csv/users.csv")
else:
    demand_csv = pd.read_csv("csv/demand.csv")
    portfolio_csv = pd.read_csv("csv/portfolios.csv")
    users_csv = pd.read_csv("csv/users.csv")

# Utility Functions
first = lambda x: x[0]
second = lambda x: x[1]
lmap = lambda f, x: list(map(f, x))

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

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
        return optim.x[0], f2(optim.x)[0] 

def create_chart(sorted_plants, day, hour, intercept_x, intercept_y):

    output_file("chart.html")

    plant_id_mw_bids = [[p.portfolio, p.id, p.capacity, b] for p, b in sorted_plants]

    sorted_p_i_m_bids = sorted(plant_id_mw_bids, key=lambda x: x[3])

    plant_MWrange_bid = [] # [[portfolio, id, xstart, xend, bid]]
    used_MW = 0

    for (p, i, mw, bid) in sorted_p_i_m_bids:
        a = [p, i, used_MW, used_MW + mw, bid]
        plant_MWrange_bid.append(a)
        used_MW += mw

    colors = ['#57BCCD', '#3976AF', '#F08636', '#529D3F', '#C63A33', '#8D6AB8', '#85594E', '#D57EBF']

    price_curve = defaultdict(list)
    for [p, i, xmin, xmax, y] in plant_MWrange_bid:
        price_curve['xs'].append([xmin, xmax])
        price_curve['ys'].append([y, y])
        price_curve['portfolio_name'].append(str(portfolio_csv.loc[portfolio_csv['portfolio'] == p, 'portfolioname'].unique()[0]))    
        price_curve['plant_name'].append(str(portfolio_csv.loc[portfolio_csv['id'] == i, 'name'].unique()[0]))
        price_curve['bid'].append(y)
        price_curve['color'].append(colors[p - 1])

    source = ColumnDataSource(price_curve)

    plot_title = "Day " + str(day) + " hour " + str(hour) + " supply and demand"

    plot = figure(title=plot_title, x_axis_label='MWH', y_axis_label='PRICE ($/MWH)', 
                  sizing_mode='stretch_width', height=400)

    supply_plot = plot.multi_line(xs='xs', ys='ys', line_width=4, line_color='color', line_alpha=0.6,
                                  hover_line_color='color', hover_line_alpha=1.0, source=source)

    plot.add_tools(HoverTool(renderers=[supply_plot], show_arrow=False, line_policy='interp', 
                             point_policy='follow_mouse', attachment='above', tooltips=[
        ('Portfolio', '@portfolio_name'),
        ('Plant', '@plant_name'),
        ('$/MWh', '@bid')
    ]))

    highest_price = plant_MWrange_bid[-1][4]
    demand_ys = [highest_price + 10., 0.]

    cur_demand_row = demand_csv[demand_csv['round'] == day][demand_csv['hour'] == hour]
    load = cur_demand_row.iloc[0]['load']
    loadslope = cur_demand_row.iloc[0]['loadslope']

    def demand_y_to_x(y): 
        return (float(y) / float(loadslope)) + load

    demand_xs = list(map(demand_y_to_x, demand_ys))

    plot.line(x=demand_xs, y=demand_ys, line_width=4, color='black')

    intercept = {'x': [float('%.2f'%(intercept_x))], 'y': [float('%.2f'%(intercept_y))]}

    intercept_plot = plot.circle('x', 'y', color='black', size=6, source=intercept)

    plot.add_tools(HoverTool(renderers=[intercept_plot], point_policy='snap_to_data', attachment='below',
                             tooltips=[
        ("", "(@x{(0.0000)} MWh, @y $/MWh)"),
    ]))


    plot.toolbar.active_drag = None


    return components(plot)



# Game Objects
class GameState(object):
    def __init__(self):
        self.noise_scale = 0
        self.auction_type = 'discrete' # can also be uniform
        self.cur_day = 1
        self.cur_hour = 1
        self.breakpoints = []
        self.demands = []
        self.prices = []
        self.carbons = []
        self.carbon_to_date = 0
        self.str_sorted_bids = []
        self.chart_div = ''
        self.chart_script = ''

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
        plant_MWrange_bid = [] # [[plant, xstart, xend, bid]]
        used_MW = 0
        # sum_xs = [sum(xs[:i+1]) for i, _ in enumerate(xs)]
        for (plant, bid) in sorted_plants:
            a = [plant, used_MW, used_MW + plant.capacity, bid]
            plant_MWrange_bid.append(a)
            used_MW += plant.capacity

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
            self.auction_type = 'discrete'
        else:
            self.auction_type = 'uniform'

    def run_hour(self, plant_bids, auto_end_day=False):
            
        plants, bids = list(map(first, plant_bids)), list(map(second, plant_bids))

        if self.cur_hour < 5: # The fifth hour allows us to see the results of hour four. It is not a true production hour.
            # tuples of (plant, bid)
            price_fn, sorted_plants, breakpoint_fn = self.construct_price_curve(plant_bids)
            self.str_sorted_bids = [] # A list of tuples (plant name, bid, produced MW)

            demand_fn = self.construct_demand_curve()

            # For each plant in sorted list:
            # Check that demand at start of plant production is above the curve
                # If not, stop production
            # Use the intermediate value theorem to figure out if demand crosses supply
                # If so, use optimization function
                    # Find intersection and stop production
                # If not, continue through the list
            
            market_demand = 0
            market_price = 0

            for plant, bid in sorted_plants:

                start_x = market_demand 
                end_x = start_x + plant.capacity # This gives us the x-coordinates of this portion of the supply curve
                
                demand_at_start_x = demand_fn(start_x)
                if demand_at_start_x < bid: 
                    # Then this segement of the supply curve is always above demand
                    # End with what market price, demand is currently set at
                    break
                else:
                    demand_at_end_x = demand_fn(end_x)
                    if (demand_at_start_x > bid and bid > demand_at_end_x):
                        # Then IVT suggests that there's an intersection in here
                        market_demand, market_price = get_intersection(price_fn, demand_fn)
                        break
                    else:
                        market_demand += plant.capacity
                        market_price = bid


            self.demands.append(float('%.2f'%(market_demand)))
            self.prices.append(float('%.2f'%(market_price)))
            self.breakpoints.append(breakpoint_fn(market_demand))
            total_activated_so_far = 0

            if self.auction_type == 'uniform':
                price_per_mwh_fn = lambda plant: price_fn(market_demand)
            elif self.auction_type == 'discrete':
                price_per_mwh_fn = lambda plant: bids[plants.index(plant)]
            else:
                raise ValueError

            hour_carbon_sum = 0
            for plant, bid in sorted_plants:

                plant.bids.append(bid) # Save a record of each plant's hourly bid history

                leftover_electricity = market_demand - total_activated_so_far
                assert leftover_electricity >= 0
                
                if leftover_electricity == 0:
                    print("All electricity used up")

                electricity_used = min(leftover_electricity, plant.capacity)
                price_per_mwh = price_per_mwh_fn(plant)
                carbon_produced = electricity_used * plant.carbon
                hour_carbon_sum += carbon_produced

                print("Plant {} used {} MWh of electricity at ${}/MWh, producing {} tons of carbon"\
                    .format(plant.name, electricity_used, price_per_mwh, carbon_produced))

                self.str_sorted_bids.append((plant.name, bid, electricity_used))

                plant.log_cost(electricity_used)
                plant.log_revenue(price_per_mwh, electricity_used)
                
                total_activated_so_far += electricity_used

            self.carbons.append(float('%.2f'%(hour_carbon_sum)))
            self.carbon_to_date += float('%.2f'%(hour_carbon_sum))

            self.chart_script, self.chart_div = create_chart(sorted_plants, self.cur_day, self.cur_hour, market_demand, market_price)

            print_players_from_plants(plants)
            reset_bids(plants)      


        self.cur_hour += 1
        if self.cur_hour == 5:
            self.end_day(plants)
        if self.cur_hour > 5:
            self.new_day(plants)

    def end_day(self, plants):
        
        # Prep hour plant data recording csv
        # csv file contains bid, production, cost, revenue, and carbon production for each hour for each plant.
        file_name = "logs/plant_hour_info_day_" + str(self.cur_day) + ".csv"
        f = open(file_name, 'w')
        headers = "portfolio,plant,bid_h1,bid_h2,bid_h3,bid_h4,mwh_h1,mwh_h2,mwh_h3,mwh_h4,carbon_h1,carbon_h2,carbon_h3,carbon_h4,revenue_h1,revenue_h2,revenue_h3,revenue_h4,cost_h1,cost_h2,cost_h3,cost_h4,cost_om\n"
        f.write(headers)
        f.close()

        for plant in plants:
            plant.transfer(file_name, day_end=True)
        print_players_from_plants(plants)
        print("Day ended")

    def new_day(self, plants):
        print("Starting new day")
        # Clean the slate
        self.demands = []
        self.prices = []
        self.carbons = []
        self.breakpoints = []
        for plant in plants:
            plant.clear()
            plant.reset()
        # Accumulate interest
        for player in set([plant.owner for plant in plants]):
            print("Accumulating interest for", player.name)
            print("Pre-interest:" + str(player.money))
            player.accumulate_interest()
            print("Post-interest:" + str(player.money))
        self.cur_hour = 1
        self.cur_day += 1
        self.switch_auction_type()


class Player(object):
    def __init__(self, name, starting_money):
        self.name = name
        self.portfolio = ""
        self.plants = []
        self.bids = {}
        self.money = float(starting_money)
        self.carbon = 0
    
    def buy_portfolio(self, portfolio_name):
        self.portfolio = portfolio_name
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
                      'o_and_m': raw_plant['fixom'],
                      'carbon': raw_plant['carbon'],
                      'portfolio': raw_plant['portfolio'],
                      'id': raw_plant['id']}
            plant = Plant(**kwargs)
            plant.owner = self
            self.plants.append(plant)
            self.bids[plant] = 0
        
    def bid(self, plant_name, bid):
        matching_plants = [plant for plant in self.plants if plant.name==plant_name]
        assert len(matching_plants) == 1
        plant = matching_plants[0]
       
        self.bids[plant] = bid
        plant.bid = bid # Current bid
    
    def accumulate_interest(self):
        self.money *= 1 + interest_rate

    def __str__(self):
        return "Portfolio: {}  |  Total Money: ${}  |  Carbon Produced: {} tons"\
                .format(self.portfolio, '%.2f'%(self.money), self.carbon)

class Plant(object):
    def __init__(self, name, capacity, cost_per_mwh, heat_rate, 
                 fuel_price, o_and_m, carbon, portfolio, id):
        # mw -> capacity; fuelcost+varom -> cost; 
        self.name = name
        self.capacity = capacity # MW
        self.cost_per_mwh = cost_per_mwh  # total var cost $/MWh
        self.heat_rate = heat_rate # MMBTU/MWh
        self.fuel_price = fuel_price # $/MMBTU
        self.o_and_m = o_and_m # $/day
        self.carbon = carbon # tons/MWh
        self.portfolio = portfolio
        self.id = id
        self.owner = None
        self.reset()
    
    def reset(self):
        # NOTE: when is o&m paid?
        self.costs = []
        self.revenues = []
        self.electricity_used = []
        self.carbon_produced = []
        self.bids = []
        print("Resetting. Name: {}, Bids: {}, Costs: {}, Revenues: {}, Electricity Used: {}, Carbon Produced: {}".format(self.name, self.bids, self.costs, self.revenues, self.electricity_used, self.carbon_produced))

        self.bid = 0

    def log_cost(self, mwh_used):
        # calculate cost for some hour
        assert 0 <= mwh_used <= self.capacity
        heat_produced = mwh_used * self.heat_rate
        cost_from_heat = heat_produced * self.fuel_price
        cost_from_var_cost = mwh_used * self.cost_per_mwh
        cost_from_carbon_tax = mwh_used * self.carbon * carbon_tax_rate
        cur_cost = cost_from_heat + cost_from_var_cost + cost_from_carbon_tax
        self.costs.append(cur_cost)
        self.electricity_used.append(mwh_used)
        self.carbon_produced.append(mwh_used * self.carbon)

    def log_revenue(self, price_per_mwh, amount_used):
        assert 0 <= amount_used <= self.capacity
        self.revenues.append(price_per_mwh * amount_used)
    
    def transfer_cost(self, day_end=False):
        if day_end:
            self.costs.append(self.o_and_m)
        total_costs = sum(self.costs)
        print("Plant {} transfering cost {:,} to {}".format(self.name, total_costs, self.owner.name))
        self.owner.money -= total_costs
    
    def transfer_revenue(self):
        total_revenues = sum(self.revenues)
        print("Plant {} transfering revenue {:,} to {}".format(self.name, total_revenues, self.owner.name))
        self.owner.money += total_revenues

    def transfer_carbon(self):
        total_carbon = sum(self.carbon_produced)
        self.owner.carbon += total_carbon
    
    def transfer(self, file_name, day_end=False):
        self.transfer_cost(day_end)
        self.transfer_revenue()
        self.transfer_carbon()

        print("Ending day. Name: {}, Bids: {}, Costs: {}, Revenues: {}, Electricity Used: {}, Carbon Produced: {}"
              .format(self.name, self.bids, self.costs, self.revenues, self.electricity_used, self.carbon_produced))
        
        # Write to csv 
        # csv file contains bid, production, cost, revenue, and carbon production for each hour for each plant.  
        plant_day_data = [self.portfolio, self.name, self.bids, self.electricity_used, self.carbon_produced, self.revenues, self.costs]
        plant_day_data = list(flatten(plant_day_data))
        with open(file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(plant_day_data)   
        f.close()


    def clear(self):
        self.revenues = []
        self.costs = []


