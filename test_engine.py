from engine import *
from utils import *
from random import randint

player_names = ['Player_1', 'Player_2', 'Player_3']
all_portfolios = [['Portfolio_1'], ['Portfolio_2'], ['Portfolio_3']]

test_hours = 2

bids = [ # days
    [ # hours
    {'Plant_1_1': 10, 'Plant_1_2': 40, 'Plant_2_1': 20, 'Plant_3_1': 30},
    {'Plant_1_1': 10, 'Plant_1_2': 40, 'Plant_2_1': 20, 'Plant_3_1': 30},
    ]
]

state = GameState()

players = [Player(name) for name in player_names]
plants = []

for player, portfolios in zip(players, all_portfolios):
    for portfolio in portfolios:
        player.buy_portfolio(portfolio)

for player in players:
    plants.extend([(plant.name, plant) for plant in player.plants])

plants = dict(plants)

for day in bids:
    for hour_bid in day:
        bids = [(plants[plant_name], bid) for plant_name, bid
                in hour_bid.iteritems()]
        state.run_hour(bids)
        print "="*10
    print make_people_table(players)
    state.end_day([plant for name, plant in plants.iteritems()])
