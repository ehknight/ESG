import dill
import pandas as pd
from collections import OrderedDict
from flask_table import Table, Col

class PlantTable(Table):
    name = Col('Plant Name')
    electricity_used = Col('Electricity Used')
    costs = Col('Costs')
    profits = Col('Profits')
    bids = Col('Current Bid')

class PeopleTable(Table):
    name = Col('Name')
    money = Col('Money')
    plants = Col('Plant')

def make_plants_table(plants):
    plants_table_constructor = []
    for plant in plants:
        plants_table_constructor.append(dict(
            name=plant.name, electricity_used = plant.electricity_used,
            costs=plant.costs, profits=plant.profits, bids=plant.bid
        ))
    plants_table = PlantTable(plants_table_constructor)
    return plants_table

def make_people_table(people):
    table_constructor = []
    for person in people:
        plants_table =  make_plants_table(person.plants)
        table_constructor.append(
            dict(name=person.name, money=person.money, plants=plants_table)
        )
    people_table = PeopleTable(table_constructor)
    return people_table.__html__()

def get_plants_from_people(people):
    plants = []
    for person in people:
        plants.extend(person.plants)
    return plants

def reset_bids(plants):
    for plant in plants:
        plant.bid = 0
        assert plant in plant.owner.bids
        plant.owner.bids[plant] = 0
    return

def get_users_and_portfolios(csv):
    user_dict = OrderedDict({'admin': {'password': 'admin', 'admin': True}})
    portfolios = dict()
    for _, row in csv.iterrows():
        info = {'password': row['Password'], 'admin': False, 
                'starting_money': row['Starting Money']}
        user_dict[row['Player Name']] = info
        portfolios[row['Player Name']] = row['Portfolio Owned']
    return user_dict, portfolios

def construct_players(users):
    from engine import Player
    player_tups = []
    for name in users:
        if users[name]['admin']: continue
        player_tups.append((name, Player(name, users[name]['starting_money'])))
    return OrderedDict(player_tups)

def buy_portfolios(players, portfolios):
    for name, portfolio in portfolios.items():
        players[name].buy_portfolio(portfolio)

def backup(save_dict, day, hour):
    for name in save_dict:
        output = open('backup/{}_day{}_hour{}.pkl'.format(name, day, hour), 'wb')
        dill.dump(save_dict[name], output)
    return