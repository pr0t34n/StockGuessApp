from flask import Flask, render_template, url_for, redirect, request, session
from flask_session import Session
from markupsafe import escape
import json
import game
import argparse
import pickle as pkl
from datetime import datetime as dt

# todo:
# playtest!
# sometimes move_forward doesn't refresh the graph properly
#	may be worth throwing another update_graph into move_forward, just to solve for this
# stoploss and, i guess, move_forward are broken in some way
# add debugging functions

# todo: finish setting up session

app = Flask(__name__)
app.config['SESSION_PERMANENT'] = True # session will not automatically expire
app.config['SESSION_TYPE'] = 'filesystem' # sessions will be stored in /flask_session
Session(app)

debug = 0

# defining variables
gv = game.GameVars()

# displays app index and passes requisite functions
@app.route('/', methods=['GET'])
def index(gv=gv):
	return render_template('index.html', gv=gv)


# gets parameters from html, as well as checked/not checked, and update variable accordingly
@app.route('/update_params/<string:req_params>', methods=['POST'])
def update_params(req_params, gv=gv):
	#print('update_params')

	# convert to dict
	req_params = json.loads(req_params)

	# sets game overlays/indicators to values received by javascript post request
	for req_k,req_v in req_params.items():

		# class
		if req_k in gv.overlays:
			for param in req_v:
				if param == 'checked':
					gv.overlays[req_k]['checked'] = req_v[param]
				else:
					for i in gv.overlays[req_k]['parameters']:
						if i['name'] == param:
							i['value'] = req_v[param]
		
		if req_k in gv.indicators:
			for param in req_v:
				if param == 'checked':
					gv.indicators[req_k]['checked'] = req_v[param]
				else:
					for i in gv.indicators[req_k]['parameters']:
						if i['name'] == param:
							i['value'] = req_v[param]

	return redirect(url_for('index'))


# called on 'refresh graph', applies tech indicators
@app.route('/graph_refresh', methods=['GET'])
def graph_refresh(gv=gv):
	#print('refresh received')

	# sometimes graph refresh happens too fast, so this refresh_start / while loop
	# ensures that the image is created before the redirect happens
	refresh_start = dt.now()
	while gv.last_updated < refresh_start:
		game.update_graph(gv, new_df=False)

	return redirect(url_for('index'))


# get next stock
@app.route('/next', methods=['GET'])
def get_next(gv=gv):
	
	game.update_graph(gv, new_df=True)

	return redirect(url_for('index'))


# todo: rename this to something a little more descriptive?
@app.route('/clear_indicators', methods=['GET'])
def clear_indicators(gv=gv):

	for k,v in gv.overlays.items():
		v['checked'] = False
	for k,v in gv.indicators.items():
		v['checked'] = False

	return redirect(url_for('graph_refresh'))


# adjusts the view parameters of the stock
@app.route('/change_timeline/<string:timeline>', methods=['POST'])
def change_timeline(timeline, gv=gv):

	print('in change_timeline, got {}'.format(timeline))

	for v in gv.game_vars['radiobutton']:
		gv.game_vars['radiobutton'][v] = False

	#'radiobutton':{'3mo':False, '6mo':True, '12mo':False},
	if timeline == '3':
		gv.game_vars['stock']['start'] = gv.game_vars['stock']['start_3m']
		gv.game_vars['radiobutton']['3mo'] = True
	if timeline == '6':
		gv.game_vars['stock']['start'] = gv.game_vars['stock']['start_6m']
		gv.game_vars['radiobutton']['6mo'] = True
	if timeline == '12':
		gv.game_vars['stock']['start'] = gv.game_vars['stock']['start_12m']
		gv.game_vars['radiobutton']['12mo'] = True

	return redirect(url_for('index'))


# moves the end date forward by specified amount
@app.route('/forward/<string:amount>', methods=['POST'])
def forward(amount, gv=gv):

	# class
	game.move_forward(amount=amount, gv=gv)

	return redirect(url_for('index'))


# clears custom values in tech indicators
@app.route('/reset_values', methods=['GET'])
def reset_values(gv=gv):

	for k,v in gv.overlays.items():
		for param in v['parameters']:
			param['value'] = param['original_value']
	for k,v in gv.indicators.items():
		for param in v['parameters']:
			param['value'] = param['original_value']

	return redirect(url_for('graph_refresh'))


# performs buy/sell action depending on whether stock is held or not
@app.route('/buy_sell', methods=['GET'])
def buy_sell(gv=gv):

	if gv.game_vars['stock']['bought']:
		gv.game_vars['stock']['bought'] = False
		game.sell(gv)
		
	else:
		gv.game_vars['stock']['bought'] = True
		game.buy(gv)

	return redirect(url_for('index'))


@app.route('/set_stoploss/<string:stoploss>', methods=['POST'])
def set_stoploss(stoploss, gv=gv):

	print('set_stoploss. received {}'.format(stoploss))
	sl = float(stoploss)

	if not gv.game_vars['stop_loss']['set']:
		gv.game_vars['stop_loss']['set'] = True
		gv.game_vars['stop_loss']['value'] = sl
	else:
		gv.game_vars['stop_loss']['set'] = False
	
	return redirect(url_for('index'))


@app.route('/stoploss_executed', methods=['POST'])
def stoploss_executed(gv=gv):
	
	gv.reset_stoploss()

	return redirect(url_for('index'))


# modified for class
@app.route('/save_and_reset', methods=['GET'])
def save_and_reset(gv=gv):

	game.save(gv)
	game.reset_game_vars(gv)

	clear_indicators()
	reset_values()

	# finalize reset
	game.update_graph(gv, new_df=True)

	return render_template('test.html')


# starts the program. note that it pulls data from saved file first
# todo: add argument parser
if __name__ == '__main__':

	timeline = 6

	parser = argparse.ArgumentParser()
	parser.add_argument('-u', action='store_true')
	parser.add_argument('-d', action='store_true')
	parser.add_argument('-dd', action='store_true')
	parser.add_argument('-t3', action='store_true')
	parser.add_argument('-t6', action='store_true')
	parser.add_argument('-t12', action='store_true')

	args = parser.parse_args()

	if args.d:
		debug = 1
	if args.dd:
		debug = 2

	if args.t3:
		timeline = 3
	elif args.t6:
		timeline = 6
	elif args.t12:
		timeline = 12

	if args.u:
		game.update_stocks()
	
	with open('./full_data.pkl', 'rb') as f:
		stock_dict = pkl.load(f)
	
	gv.load_stock_dict(stock_dict)
	gv.set_timeline(timeline)

	game.update_graph(gv, new_df=True)
	app.run(debug=True)





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~