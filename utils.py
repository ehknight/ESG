from flask_table import Table, Col

class PlantTable(Table):
    name = Col('Plant Name')
    electricity_used = Col('Electricity Used')
    costs = Col('Costs')
    profits = Col('Profits')
    bids = Col('Bids')

class PeopleTable(Table):
    name = Col('Name')
    money = Col('Money')
    plants = Col('Plants')

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