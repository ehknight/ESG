from flask import Flask, render_template, flash, request, session, send_from_directory
from wtforms import Form, FloatField, TextField, TextAreaField, validators, StringField, SubmitField, SelectField, FieldList, FormField
from flask_login import login_required, LoginManager, current_user
from collections import OrderedDict
import flask_login
import flask
import dill
import click
from engine import *
from utils import *
import logging

########## SETUP ##########

DEBUG = False
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
login_manager = LoginManager(app)
app.config['SECRET_KEY'] = 'bnoeuzoeqpv'
state = GameState()

users, portfolios = get_users_and_portfolios(users_csv)
players = construct_players(users)
buy_portfolios(players, portfolios)

########## USER MANAGEMENT ##########

class User(flask_login.UserMixin):
    pass

@login_manager.user_loader
def user_loader(email):
    if email not in users:
        return
    user = User()
    user.id = email
    return user

@login_manager.request_loader
def request_loader(request):
    email = request.form.get('email')
    if email not in users:
        return

    user = User()
    user.id = email

    # DO NOT ever store passwords in plaintext and always compare password
    # hashes using constant-time comparison!
    user.is_authenticated = request.form['password'] == users[email]['password']

    return user

@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'GET':
        return render_template('login.html')

    email = flask.request.form['email']
    try:
        users[email]
    except KeyError: # email doesn't exist
        return flask.redirect(flask.url_for('login'))

    if flask.request.form['password'] == users[email]['password']:
        user = User()
        user.id = email
        flask_login.login_user(user)
        if current_user.id == 'admin':
            return flask.redirect(flask.url_for('adminview'))
        else:
            return flask.redirect(flask.url_for('playerview'))

    return flask.redirect(flask.url_for('login'))


@app.route('/logout')
def logout():
    flask_login.logout_user()
    return render_template('logout.html')

@login_manager.unauthorized_handler
def unauthorized_handler():
    return 'Incorrect credentials. <a href="/login">Back to login</a>'

@app.route('/')
def mainhtml():
    return flask.redirect(flask.url_for('playerview'))

########## FORMS ETC ##########

class BidForm(Form):
    plant_name = StringField('plant_name')
    bid = FloatField('bid')

class PortfolioForm(Form):
    title = 'title'
    plantbids = FieldList(FormField(BidForm))

@login_required
def construct_form(current_user):
    global players
    class ReusableForm(Form):
        plant_names = [(p.name, p.name) for p in players[current_user.id].plants]
        plant = SelectField('Plants', choices=plant_names)
        bid = FloatField('Bid')
    return ReusableForm
    

########## WEB CONTROL ##########
@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

@login_required
@app.route('/admin', methods=['GET', 'POST'])
def adminview():
    global state
    try: current_user.id == 'test'
    except: return flask.redirect(flask.url_for('login'))

    if current_user.id != 'admin':
        return flask.redirect(flask.url_for('login'))
    
    if request.method == 'POST':
        print("Advancing to next hour")
        player_objs = [player for name, player in players.items()]
        all_plants = get_plants_from_people(player_objs)
        bids = [(plant, plant.bid) for plant in all_plants]
        state.run_hour(bids)
        backup({'state': state, 'players': players}, 
                state.cur_day, state.cur_hour)
    
    tables = [(name, str(player), make_plants_table(player.plants)) 
              for name, player in players.items()]

    return render_template('admin.html', player_info=tables, auction_type=state.auction_type,
                            day=state.cur_day, hour=state.cur_hour, demands=state.demands, 
                            prices=state.prices, carbons=state.carbons, carbon_to_date=state.carbon_to_date,
                            breakpoints=state.breakpoints, sorted_bids=state.str_sorted_bids, 
                            chart_div=state.chart_div, chart_script=state.chart_script)

@login_required
@app.route("/player", methods=['GET', 'POST'])
def playerview():
    try: current_user.id == 'test'
    except: return flask.redirect(flask.url_for('login'))
    if current_user.id == 'admin':
        return flask.redirect(flask.url_for('login'))
    else:
        user = current_user
        player = players[user.id]

    form = PortfolioForm()
    form.title = str(user.id)
    for plant in player.plants:
        bid_form = BidForm()
        bid_form.plant_name = plant.name
        bid_form.bid = plant.bid
        form.plantbids.append_entry(bid_form)

    if request.method == 'POST':

        bids = []

        request_dict = request.form.to_dict(flat=True)

        if form.validate():
            for [_, bid] in request_dict.items():
                bids.append(bid)
            plant_bid_list = zip(player.plants, bids)

            for [plant, bid] in plant_bid_list:
                try: # HACK: FloatField validation broken? So we're catching it here.  
                    bid = float(bid)
                except:
                    bid = 0.
                bid = max(0., min(bid, 500.))
                flash('Bid {} on plant {}'.format(bid, plant.name))
                player.bid(plant.name, bid)
            return flask.redirect(flask.url_for('playerview'))
        else:
            flash(form.errors)
            flash('There are errors in the form. Please double check')

        print(dict(request.form))

    table = make_plants_table(player.plants)
    kwargs = {
        'table':table, 'form':form, 'name': user.id,
        'day': state.cur_day, 'hour': state.cur_hour,
        'auction_type': state.auction_type, 'player_info': str(player),
        'demands': state.demands, 'prices': state.prices, 
        'carbons': state.carbons, 'carbon_to_date': state.carbon_to_date,
        'breakpoints': state.breakpoints, 'sorted_bids': state.str_sorted_bids, 
        'chart_div': state.chart_div, 'chart_script': state.chart_script
    }

    return render_template('player.html', **kwargs)

@click.command()
@click.option('--state_backup')
@click.option('--player_backup')
def main(state_backup, player_backup):
    global state, players
    if state_backup:
        print("LOADED STATE")
        state = dill.load(open(state_backup))
    if player_backup:
        players = dill.load(open(player_backup))
        print("LOADED PLAYERS")
    app.run(debug=DEBUG, threaded=True, port=80, host='0.0.0.0')

main()
