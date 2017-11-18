from engine import *
from random import randint

player_names = ['E', 'S', 'G']
all_portfolios = [['Nash', 'Ostrom'], ['Modigliani'], ['Vickrey']]
bids = []
state = GameState()

players = [Player(name) for name in player_names]
for player, portfolios in zip(players, all_portfolios):
    for portfolio in portfolios:
        player.buy_portfolio(portfolio)

for player in players:
    for plant in player.plants:
        bids.append((plant, randint(0, 1000000)))

state.run_hour(bids)