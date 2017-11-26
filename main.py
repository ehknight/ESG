from flask import Flask, render_template, flash, request, session
from wtforms import Form, FloatField, TextField, TextAreaField, validators, StringField, SubmitField, SelectField
from flask_login import login_required, LoginManager, current_user
from collections import OrderedDict
import flask_login
import flask
import dill
from engine import *
from utils import *

########## SETUP ##########

DEBUG = True
app = Flask(__name__)
login_manager = LoginManager(app)
app.config['SECRET_KEY'] = 'pastaelephantgreenleafshoe'
state = GameState()

users = OrderedDict([('admin', {'password': 'admin', 'admin': True}),
        ('Player_1', {'password': 'player', 'admin': False}),
        ('Player_2', {'password': 'player', 'admin': False}),
        ('Player_3', {'password': 'player', 'admin': False}),
])

players = OrderedDict([(key, Player(key)) for key in users if key != 'admin'])

def backup():
    global state, players
    to_save = {'state': state, 'players': players}
    for name in to_save:
        output = open('backup_{}_day{}_hour{}.pkl'\
                .format(name, state.cur_day, state.cur_hour), 'wb')
        dill.dump(to_save[name], output)
    return

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
        return flask.redirect(flask.url_for('playerview'))

    return flask.redirect(flask.url_for('playerview'))

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

########## TESTING ##########

players['Player_1'].buy_portfolio('Portfolio_1')
players['Player_2'].buy_portfolio('Portfolio_2')
players['Player_3'].buy_portfolio('Portfolio_3')

########## FORMS ETC ##########

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
        print "Advancing to next hour"
        player_objs = [player for name, player in players.iteritems()]
        all_plants = get_plants_from_people(player_objs)
        bids = [(plant, plant.bid) for plant in all_plants]
        state.run_hour(bids)
        backup()
    
    tables = [(name, make_plants_table(player.plants)) 
              for name, player in players.iteritems()]

    return render_template('admin.html', player_info=tables,
                            day=state.cur_day, hour=state.cur_hour)

@login_required
@app.route("/player", methods=['GET', 'POST'])
def playerview():
    try: current_user.id == 'test'
    except: return flask.redirect(flask.url_for('login'))
    if current_user.id == 'admin':
        return flask.redirect(flask.url_for('adminview'))
    else:
        user = current_user
        player = players[user.id]
    form = construct_form(user)(request.form)
 
    print form.errors
    if request.method == 'POST':
        plant_name = request.form['plant']
        bid = request.form['bid']
 
        if form.validate():
            # Save the comment here.
            bid = float(bid)
            flash('Bid {} on plant {}'.format(bid, plant_name))
            player.bid(plant_name, bid)
        else:
            flash(form.errors)
            flash('There are errors in the form. Please double check')
    table = make_plants_table(player.plants)
    kwargs = {
        'table':table, 'form':form, 'name': user.id,
        'day': state.cur_day, 'hour': state.cur_hour
    }

    return render_template('player.html', **kwargs)

if __name__ == '__main__':
    app.run(debug=DEBUG)