import pandas as pd 
import numpy as np 
import os 
import random 
import pickle as pkl 
import re 
import requests 
import json
import locale

import mplfinance as mpf 
import matplotlib
from matplotlib import pyplot as plt 
from matplotlib.ticker import ScalarFormatter 
matplotlib.use('agg')

import talib 

from datetime import datetime as dt
from scipy import signal 
from yahoo_fin.stock_info import * # calls: df = get_data('SYM', start_date='yyyy-mm-dd')

internal_stock_dict = {}

locale.setlocale(locale.LC_ALL, '')


class GameVars:

	def __init__(self):
		self._make_indicators()
		self._make_overlays()
		self._make_game_vars()

		self.last_updated = dt.now()

	def load_stock_dict(self, stock_dict):
		self.stock_dict = stock_dict

	def _make_game_vars(self):
		# note: internal-only values do not have 'name' fields
		self.game_vars = {
			'starting_amt':{'name':'Starting Amount:', 'id':'starting_amt', 'value':'$100,000', 'num_value':100000},
			'current_funds':{'name':'Available Funds:', 'id':'current_funds', 'value':'$100,000', 'num_value':100000},
			'num_seen':{'name':'Num Seen:', 'id':'num_seen', 'value':0},
			'num_bought':{'name':'Num Bought:', 'id':'num_bought', 'value':0},
			'held_amt':{'name':'Amount Holding:', 'id':'held_amt', 'value':0},
			'wins':{'name':'Wins:', 'id':'wins', 'value':0},
			'losses':{'name':'Losses:', 'id':'losses', 'value':0},
			'win_pct':{'name':'Win Pct:', 'id':'win_pct', 'value':0.0},
			'av_days_held':{'name':'Av Days Held:', 'id':'av_days_held', 'value':0, 'array':[]},
			'rolling_ret_pct':{'name':'Rolling Return Pct:', 'id':'rolling_ret_pct','value':0.0},
			'ret_pct':{'name':'Last Return Pct:', 'id':'ret_pct', 'value':0.0},
			'current_close':{'name':'Current Close:', 'id':'current_close', 'value':'$0', 'num_value':0.0},
			'purchase_price':{'name':'Purchase Price:', 'id':'purchase_price', 'value':'$0', 'num_value':0.0},
			'current_days_held':{'value':0},
			'stop_loss':{'value':0.0, 'triggered':False, 'date_executed':'', 'price_executed':0.0, 'set':False},
			'stock':self._make_gv_stock(),
			'radiobutton':{'3mo':False, '6mo':True, '12mo':False},
			'timeline':{'value':6},
		}

	def _make_gv_stock(self):
		stock = {
			'df':None,'ticker':'','df_index_arr':'','bought':False,
			'start':'','start_3m':'','start_6m':'','start_12m':'', 
			'end':'','yesterday':'','end_i':'',
		}
		return stock

	def _make_overlays(self):
		self.overlays = {
			'sma1':self._make_ti('sma1', [{'name':'period', 'col':6, 'value':5, 'original_value':5}]),
			'sma2':self._make_ti('sma2', [{'name':'period', 'col':6, 'value':20, 'original_value':20}]),
			'sma3':self._make_ti('sma3', [{'name':'period', 'col':6, 'value':200, 'original_value':200}]),
			'ema1':self._make_ti('ema1', [{'name':'period', 'col':6, 'value':5, 'original_value':5}]),
			'ema2':self._make_ti('ema2', [{'name':'period', 'col':6, 'value':15, 'original_value':15}]),
			'ema3':self._make_ti('ema3', [{'name':'period', 'col':6, 'value':30, 'original_value':30}]),
			'psar':self._make_ti('psar', [{'name':'max', 'col':3, 'value':0.02, 'original_value':0.02}, {'name':'step', 'col':3, 'value':0.2, 'original_value':0.02}]),
			'bollinger':self._make_ti('bollinger', [{'name':'period', 'col':3, 'value':20, 'original_value':20}, {'name':'stdev', 'col':3, 'value':2, 'original_value':2}]),
		}

	def _make_indicators(self):
		self.indicators = {
			'macd':self._make_ti('macd', [{'name':'fast', 'col':2, 'value':12, 'original_value':12}, {'name':'slow', 'col':2, 'value':25, 'original_value':25}, {'name':'signal', 'col':2, 'value':9, 'original_value':9}]),
			'rsi':self._make_ti('rsi', [{'name':'period', 'col':6, 'value':14, 'original_value':14}]),
			'adx':self._make_ti('adx', [{'name':'period', 'col':6, 'value':14, 'original_value':14}]),
			'stoch':self._make_ti('stoch', [{'name':'fastk', 'col':2, 'value':14, 'original_value':14}, {'name':'slowk', 'col':2, 'value':3, 'original_value':3}, {'name':'slowd', 'col':2, 'value':3, 'original_value':3}]),
			'stdev':self._make_ti('stdev', [{'name':'period', 'col':6, 'value':10, 'original_value':10}]),
			'williams':self._make_ti('williams', [{'name':'period', 'col':6, 'value':14, 'original_value':14}]),
			'cci':self._make_ti('cci', [{'name':'period', 'col':6, 'value':20, 'original_value':20}]),
			'aroon':self._make_ti('aroon', [{'name':'period', 'col':6, 'value':25, 'original_value':25}]),
			'obv':self._make_ti('obv', []),
			'accumdist':self._make_ti('accumdist', []),
		}

	# clears stoploss, radiobutton, purchase price, and stock data. called from next()
	def _clear_vars(self):
		#'stop_loss':{'value':0.0, 'triggered':False, 'date_executed':'', 'price_executed':0.0, 'set':False},
		self.reset_stoploss()
		self.reset_radiobutton()
		self.game_vars['purchase_price']['num_value'] = 0.0
		self.game_vars['purchase_price']['value'] = locale.currency(self.game_vars['purchase_price']['num_value'])
		#self.game_vars['stock'] = {'df':None, 'start':'', 'start_3m':'', 'start_6m':'', 'start_12m':'', 'end':'', 'ticker':'', 'df_index_arr':'', 'bought':False}
		self.game_vars['stock'] = self._make_gv_stock()

	# simple internal function to generate tech indicator details for web processing
	def _make_ti(self, name, params):
		d = {'name':name, 'id':name, 'checked':False, 'parameters':params}
		return d

	def _unset_radiobutton(self):
		self.game_vars['radiobutton']['3mo'] = False
		self.game_vars['radiobutton']['6mo'] = False
		self.game_vars['radiobutton']['12mo'] = False

	def set_timeline(self, timeline):
		#print('received timeline: {}'.format(timeline))
		self.game_vars['timeline']['value'] = int(timeline)

	# generates or appends to a csv file to record stats from last run
	def save(self, filename):

		date = dt.now()
		datestr = f"{date.year}-{date.month}-{date.day} {date.hour}:{date.minute}"
		funds = "{:.2f}".format(self.game_vars['current_funds']['num_value'])
		num_seen = self.game_vars['num_seen']['value']
		num_bought = self.game_vars['num_bought']['value']
		wins = self.game_vars['wins']['value']
		losses = self.game_vars['losses']['value']
		win_pct = "{:.2f}".format(self.game_vars['win_pct']['value'] * 100)
		av_days_held = "{:.2f}".format(self.game_vars['av_days_held']['value'])
		end_return = "{:.2f}".format(self.game_vars['rolling_ret_pct']['value'] * 100 - 100)

		write_str = f"{datestr},{funds},{num_seen},{num_bought},{wins},{losses},{win_pct},{av_days_held},{end_return}\n"

		if not os.path.isfile(filename):
			# if the file doesn't exist, first make a header
			header = "date,ending_funds,seen,bought,wins,losses,win_percent,avg_days_held,return_percent\n"
			with open(filename, 'w+') as f:
				f.write(header)

		with open(filename, 'a') as f:
			f.write(write_str)

	# resets all game_vars to default values
	def reset_game_vars(self):
		self._make_game_vars()

	# resets stoploss values
	def reset_stoploss(self):
		self.game_vars['stop_loss']['value'] = 0.0
		self.game_vars['stop_loss']['triggered'] = False
		self.game_vars['stop_loss']['date_executed'] = ''
		self.game_vars['stop_loss']['price_executed'] = 0.0
		self.game_vars['stop_loss']['set'] = False

	# sets timeline back to default
	def reset_radiobutton(self):
		self.game_vars['radiobutton']['3mo'] = False
		self.game_vars['radiobutton']['6mo'] = True
		self.game_vars['radiobutton']['12mo'] = False


# calls class method which resets all game vars (not indicators/overlays) to default values
def reset_game_vars(gv):
	gv.reset_game_vars()


# buy function rewrite
def buy(gv):

	buy_price = gv.game_vars['stock']['df']['close'].loc[gv.game_vars['stock']['end']]
	gv.game_vars['num_bought']['value'] += 1
	gv.game_vars['purchase_price']['num_value'] = buy_price

	gv.game_vars['held_amt']['value'] = int(gv.game_vars['current_funds']['num_value'] / gv.game_vars['purchase_price']['num_value'])
	gv.game_vars['current_funds']['num_value'] = gv.game_vars['current_funds']['num_value'] - (gv.game_vars['held_amt']['value'] * gv.game_vars['purchase_price']['num_value'])
	gv.game_vars['current_funds']['value'] = locale.currency(gv.game_vars['current_funds']['num_value'], grouping=True)
	gv.game_vars['purchase_price']['value'] = locale.currency(gv.game_vars['purchase_price']['num_value'])


# sell function rewrite
def sell(gv, sell_amount=None):
	
	if sell_amount:
		sold_price = sell_amount
	else:
		sold_price = gv.game_vars['stock']['df']['close'].loc[gv.game_vars['stock']['end']]

	bought_price = gv.game_vars['purchase_price']['num_value']
	amount_held = gv.game_vars['held_amt']['value']
	ret_pct = float("{:.2f}".format(sold_price/bought_price))

	ret_amt = sold_price * amount_held
	ret_amt_minus_bought = (sold_price - bought_price) * amount_held

	if ret_amt_minus_bought > 0:
		gv.game_vars['wins']['value'] += 1
	elif ret_amt_minus_bought < 0:
		gv.game_vars['losses']['value'] += 1

	win_pct = gv.game_vars['wins']['value'] / gv.game_vars['num_bought']['value']
	win_pct = "{:.2f}".format(win_pct)
	gv.game_vars['win_pct']['value'] = float(win_pct)

	gv.game_vars['av_days_held']['array'].append(gv.game_vars['current_days_held']['value'])
	gv.game_vars['av_days_held']['value'] = float("{:.2f}".format(np.mean(gv.game_vars['av_days_held']['array'])))
	gv.game_vars['current_days_held']['value'] = 0

	#print(f"uninvested[num_val]: {game_vars['current_funds']['num_value'] }, ret_amt: {ret_amt}")
	final_amt = gv.game_vars['current_funds']['num_value'] + ret_amt
	gv.game_vars['current_funds']['num_value'] = final_amt
	gv.game_vars['current_funds']['value'] = locale.currency(final_amt, grouping=True)

	#'rolling_ret_pct':{'name':'Rolling Return Pct:', 'id':'rolling_ret_pct','value':0.0},
	gv.game_vars['rolling_ret_pct']['value'] = float("{:.2f}".format((gv.game_vars['current_funds']['num_value']/gv.game_vars['starting_amt']['num_value'])*100 - 100))

	gv.game_vars['held_amt']['value'] = 0
	gv.game_vars['stock']['bought'] = False


def check_stoploss(gv, new_end_i, end_i):
	#print('check_stoploss')
	stop_loss = float(gv.game_vars['stop_loss']['value'])
	df_index_arr = gv.game_vars['stock']['df_index_arr']
	df = gv.game_vars['stock']['df']

	# iterate through the date diff until found stoploss trigger
	for i in range(new_end_i - end_i):
		date = df_index_arr[end_i + i]
		print('stoploss: checking date: {}'.format(date))
		#print('move_forward: stoploss check date: {}'.format(date))
		#print('values: close: {:.2f}, low: {:.2f}, open: {:.2f}'.format(df['close'].loc[date], df['low'].loc[date], df['open'].loc[date]))

		if df['close'].loc[date] < stop_loss or df['low'].loc[date] < stop_loss or df['open'].loc[date] < stop_loss:
			#print('move_forward: stoploss less than a value')

			if gv.game_vars['stop_loss']['triggered'] == False:

				gv.game_vars['stop_loss']['triggered'] = True
				gv.game_vars['stop_loss']['date_executed'] = date # for sending an alert
				
				# if it opened under the stoploss, that's the sell value
				if df['open'].loc[date] < stop_loss:
					gv.game_vars['stop_loss']['price_executed'] = float("{:.2f}".format(df['open'].loc[date]))

				# otherwise, assume it sold exactly at stoploss value
				else:
					gv.game_vars['stop_loss']['price_executed'] = gv.game_vars['stop_loss']['value']

				sell(gv, gv.game_vars['stop_loss']['price_executed'])
				print('sold stock at stoploss value: {:.2f}'.format(gv.game_vars['stop_loss']['price_executed']))


# move forward rewrite
def move_forward(amount, gv):
	#print('move_forward')

	df = gv.game_vars['stock']['df']

	if gv.game_vars['stock']['bought']:
		gv.game_vars['current_days_held']['value'] += int(amount)

	df_index_arr = gv.game_vars['stock']['df_index_arr']

	# old_end is a date, end_i is the position in the index
	old_end = gv.game_vars['stock']['end']
	end_i = np.where(np.in1d(df_index_arr, old_end))[0][0]

	if end_i + int(amount) > len(df_index_arr) - 1:
		new_end_i = len(df_index_arr)
	else:
		new_end_i = end_i + int(amount)

	# got index value above, now get date value
	new_end = df_index_arr[new_end_i]
	new_yest = df_index_arr[new_end_i-1]

	#print('current end date: {}'.format(gv.game_vars['stock']['end']))
	#print('new end date: {}'.format(new_end))

	# check for stop losses
	if gv.game_vars['stop_loss']['set'] == True:
		check_stoploss(gv=gv, new_end_i=new_end_i, end_i=end_i)

	# set end date to new end date
	gv.game_vars['stock']['end'] = new_end
	gv.game_vars['stock']['yesterday'] = new_yest
	gv.game_vars['current_close']['num_value'] = gv.game_vars['stock']['df']['close'].loc[gv.game_vars['stock']['end']]
	gv.game_vars['current_close']['value'] = locale.currency(gv.game_vars['current_close']['num_value'])


# clear vars rewrite, now in the class
def clear_vars(gv):
	gv._clear_vars()


# rewrite of update graph to reference class
def update_graph(gv, addplot=[], new_df=True):

	#print('in update_graph')

	# kill the plot before spawning a new one
	plt.close()

	# load a new df from random
	if new_df:
		print('loading random stock')

		# resets radiobutton, stoploss, purchase price, game_vars[stock]
		clear_vars(gv)

		stock_chosen = False

		# in addition to getting a random df,
		# get random start and end, where there is at least a year of prior data
		# and a few months of post data
		while not stock_chosen:
			#print('not stock chosen')
			symbol = random.choice(list(gv.stock_dict.keys()))
			df = gv.stock_dict[symbol]['df']

			if len(df) > 500:
				end = random.choice(df.iloc[:-50].index.values)

				df_index_arr = df.index.values
				end_i = np.where(np.in1d(df_index_arr, end))[0][0]

				gv.game_vars['stock']['end_i'] = end_i
				yest = end_i - 1
				yest = df_index_arr[yest]
				gv.game_vars['stock']['yesterday'] = yest

				#print('end_i: {}'.format(end_i))
				start_i = end_i - 250

				if start_i > 0:
					start_3m_i = end_i - 60
					start_6m_i = end_i - 120
					start_12m_i = end_i - 240

					start_3m = df_index_arr[start_3m_i]
					start_6m = df_index_arr[start_6m_i]
					start_12m = df_index_arr[start_12m_i]

					gv.game_vars['stock']['start_3m'] = start_3m
					gv.game_vars['stock']['start_6m'] = start_6m
					gv.game_vars['stock']['start_12m'] = start_12m

					# set timeline
					gv._unset_radiobutton()

					if gv.game_vars['timeline']['value'] == 3:
						#print('setting timeline: 3mo')
						start_time = start_3m
						gv.game_vars['radiobutton']['3mo'] = True

					if gv.game_vars['timeline']['value'] == 6:
						#print('setting timeline: 6mo')
						start_time = start_6m
						gv.game_vars['radiobutton']['6mo'] = True

					if gv.game_vars['timeline']['value'] == 12:
						#print('setting timeline: 12mo')
						start_time = start_12m
						gv.game_vars['radiobutton']['12mo'] = True

					gv.game_vars['stock']['start'] = start_time
					gv.game_vars['stock']['end'] = end
					gv.game_vars['stock']['df_index_arr'] = df_index_arr
					stock_chosen = True

		gv.game_vars['stock']['ticker'] = symbol
		gv.game_vars['stock']['df'] = df
		gv.game_vars['num_seen']['value'] += 1

	df = gv.game_vars['stock']['df']

	gv.game_vars['current_close']['num_value'] = df['close'].loc[gv.game_vars['stock']['end']]
	gv.game_vars['current_close']['value'] = locale.currency(gv.game_vars['current_close']['num_value'])

	# last return percent
	last_ret_pct = ((df['close'].loc[gv.game_vars['stock']['end']] / df['close'].loc[gv.game_vars['stock']['yesterday']]) * 100) - 100
	last_ret_pct = float("{:.2f}".format(last_ret_pct))
	gv.game_vars['ret_pct']['value'] = last_ret_pct

	print('current end date: {}'.format(gv.game_vars['stock']['end']))

	plot_area = df.loc[gv.game_vars['stock']['start']:gv.game_vars['stock']['end']]

	# get any addplots
	addplot = get_addplots(df, gv)

	# set colors and limits
	mc = mpf.make_marketcolors(up='g', down='r', volume='in')
	s = mpf.make_mpf_style(marketcolors=mc)

	ylow = .9 * min(plot_area['low'])
	yhigh = 1.1 * max(plot_area['high'])
	ylim = (ylow, yhigh)

	# perform plot
	fig, axlist = mpf.plot(
		plot_area,
		type='candle',
		style=s,
		ylim=ylim,
		warn_too_much_data=len(plot_area)+20,
		figscale=1.05,
		volume=True,
		addplot=addplot,
		returnfig=True,
		title=gv.game_vars['stock']['ticker'],
	)

	# set log scale
	ax = axlist[0]
	ax.set_yscale('log')
	ax.yaxis.set_major_formatter(ScalarFormatter())
	ax.yaxis.set_minor_formatter(ScalarFormatter())

	# hides xaxis labels
	ax.set_xticks([])

	#print('saving figure. figure: {}'.format(fig))
	fig.savefig('./static/stock_img_tmp.png', bbox_inches='tight')
	gv.last_updated = dt.now()


# determines if overlay is checked and returns dict for each active overlay with value
def eval_overlays(df, gv):
	#print('eval_overlays')
	ap_overlays = {}

	for o in gv.overlays:
		#print('overlays[o]: {}'.format(overlays[o]))
		if gv.overlays[o]['checked'] == True:
			if o == 'sma1':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':	
						timeperiod = int(v['value'])
				sma1 = talib.SMA(df['close'], timeperiod=timeperiod)
				ap_overlays['sma1'] = sma1
			
			if o == 'sma2':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':	
						timeperiod = int(v['value'])
				sma2 = talib.SMA(df['close'], timeperiod=timeperiod)
				ap_overlays['sma2'] = sma2
			
			if o == 'sma3':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':	
						timeperiod = int(v['value'])
				sma3 = talib.SMA(df['close'], timeperiod=timeperiod)
				ap_overlays['sma3'] = sma3

			if o == 'ema1':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':	
						timeperiod = int(v['value'])
				ema1 = talib.EMA(df['close'], timeperiod=timeperiod)
				ap_overlays['ema1'] = ema1

			if o == 'ema2':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':	
						timeperiod = int(v['value'])
				ema2 = talib.EMA(df['close'], timeperiod=timeperiod)
				ap_overlays['ema2'] = ema2

			if o == 'ema3':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':	
						timeperiod = int(v['value'])
				ema3 = talib.EMA(df['close'], timeperiod=timeperiod)
				ap_overlays['ema3'] = ema3

			#'psar':make_ti('psar', [{'name':'max', 'col':3, 'value':0.02, 'original_value':0.02}, {'name':'step', 'col':3, 'value':0.2, 'original_value':0.02}]),
			if o == 'psar':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'max':
						psar_max = float(v['value'])
					if v['name'] == 'step':
						psar_step = float(v['value'])

				psar = talib.SAR(df['high'], df['low'], acceleration=psar_step, maximum=psar_max)
				ap_overlays['psar'] = psar

			#'bollinger':make_ti('bollinger', [{'name':'period', 'col':3, 'value':20, 'original_value':20}, {'name':'stdev', 'col':3, 'value':2, 'original_value':2}]),
			if o == 'bollinger':
				for v in gv.overlays[o]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
					if v['name'] == 'stdev':
						stdev = int(v['value'])
				upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=timeperiod, nbdevup=stdev, nbdevdn=stdev)
				ap_overlays['bollinger_upperband'] = upperband
				ap_overlays['bollinger_lowerband'] = lowerband
				ap_overlays['bollinger_middleband'] = middleband


		else:
			if o == 'sma1':
				ap_overlays['sma1'] = None
			if o == 'sma2':
				ap_overlays['sma2'] = None
			if o == 'sma3':
				ap_overlays['sma3'] = None
			if o == 'ema1':
				ap_overlays['ema1'] = None
			if o == 'ema2':
				ap_overlays['ema2'] = None
			if o == 'ema3':
				ap_overlays['ema3'] = None
			if o == 'psar':
				ap_overlays['psar'] = None
			if o == 'bollinger':
				ap_overlays['bollinger_lowerband'] = None
				ap_overlays['bollinger_upperband'] = None
				ap_overlays['bollinger_middleband'] = None

	return ap_overlays


# determines if indicator is checked 
# returns dict for each active indicator with value, along with graph panel position
def eval_indicators(df, gv):
	#print('eval_indicators')
	ap_indicators = {}

	ind_panel_manager = {}
	ind_panel_manager['macd'] = 0
	ind_panel_manager['stoch'] = 0
	ind_panel_manager['aroon'] = 0
	ind_panel_counter = 2

	for i in gv.indicators:
		if gv.indicators[i]['checked'] == True:

			# 'macd':make_ti('macd', [{'name':'fast', 'col':2, 'value':12, 'original_value':12}, {'name':'slow', 'col':2, 'value':25, 'original_value':25}, {'name':'signal', 'col':2, 'value':9, 'original_value':9}]),
			if i == 'macd':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'fast':
						fast = int(v['value'])
					if v['name'] == 'slow':
						slow = int(v['value'])
					if v['name'] == 'signal':
						signal = int(v['value'])
				macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
				ap_indicators['macd'] = macd 
				ap_indicators['macdsignal'] = macdsignal
				ap_indicators['macdhist'] = macdhist

			# 'rsi':make_ti('rsi', [{'name':'period', 'col':6, 'value':14, 'original_value':14}]),
			if i == 'rsi':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
				rsi = talib.RSI(df['close'], timeperiod=timeperiod)
				ap_indicators['rsi'] = rsi 

			# 'adx':make_ti('adx', [{'name':'period', 'col':6, 'value':14, 'original_value':14}]),
			if i == 'adx':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
				adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=timeperiod)
				ap_indicators['adx'] = adx

			# 'stoch':make_ti('stoch', [{'name':'fastk', 'col':2, 'value':14, 'original_value':14}, {'name':'slowk', 'col':2, 'value':3, 'original_value':3}, {'name':'slowd', 'col':2, 'value':3, 'original_value':3}]),
			if i == 'stoch':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'fastk':
						fastk_p = int(v['value'])
					if v['name'] == 'slowk':
						slowk_p = int(v['value'])
					if v['name'] == 'slowd':
						slowd_p = int(v['value'])
				slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=fastk_p, slowk_period=slowk_p, slowd_period=slowd_p)
				ap_indicators['stoch_slowk'] = slowk 
				ap_indicators['stoch_slowd'] = slowd 

			# 'williams':make_ti('williams', [{'name':'period', 'col':6, 'value':14, 'original_value':14}]),
			if i == 'williams':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
				willr = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
				ap_indicators['willr'] = willr

			# 'stdev':make_ti('stdev', [{'name':'period', 'col':6, 'value':10, 'original_value':10}]),
			if i == 'stdev':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
				stdev = talib.STDDEV(df['close'], timeperiod=timeperiod)
				ap_indicators['stdev'] = stdev

			# 'cci':make_ti('cci', [{'name':'period', 'col':6, 'value':20, 'original_value':20}]),
			if i == 'cci':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
				cci = talib.CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
				ap_indicators['cci'] = cci 

			# 'aroon':make_ti('aroon', [{'name':'period', 'col':6, 'value':25, 'original_value':25}]),
			if i == 'aroon':
				for v in gv.indicators[i]['parameters']:
					if v['name'] == 'period':
						timeperiod = int(v['value'])
				aroondown, aroonup = talib.AROON(df['high'], df['low'], timeperiod=timeperiod)
				ap_indicators['aroon_up'] = aroonup
				ap_indicators['aroon_down'] = aroondown

			# 'obv':make_ti('obv', []),
			if i == 'obv':
				obv = talib.OBV(df['close'], df['volume'])
				ap_indicators['obv'] = obv

			# 'accumdist':make_ti('accumdist', []),
			if i == 'accumdist':
				adl = talib.AD(df['high'], df['low'], df['close'], df['volume'])
				ap_indicators['accumdist'] = adl 


		else:
			if i == 'macd':
				ap_indicators['macd'] = None
				ap_indicators['macdsignal'] = None
				ap_indicators['macdhist'] = None

				if ind_panel_manager['macd'] > 0:
					ind_panel_manager['macd'] = 0
				if ind_panel_counter > 2:
					ind_panel_counter -= 1
			
			if i == 'rsi':
				ap_indicators['rsi'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'adx':
				ap_indicators['adx'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'stoch':
				ap_indicators['stoch_slowk'] = None
				ap_indicators['stoch_slowd'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'williams':
				ap_indicators['willr'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'stdev':
				ap_indicators['stdev'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'cci':
				ap_indicators['cci'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'aroon':
				ap_indicators['aroon_up'] = None
				ap_indicators['aroon_down'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'obv':
				ap_indicators['obv'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

			if i == 'accumdist':
				ap_indicators['accumdist'] = None
				if ind_panel_counter > 2:
					ind_panel_counter -= 1

	return ap_indicators, ind_panel_manager, ind_panel_counter


# get additional plots requests from variables
def get_addplots(df, gv):
	#print('get_addplots')
	
	# cut df to stop area to avoid functions performing predictive actions
	df = df[:gv.game_vars['stock']['end']]
	
	# will contain a list of addplots generated by mpf.make_addplot()
	ap_arr = []

	# do the thing that makes the frame
	def _make_frame(i,d):
		#print('make_frame')

		tmp_df = pd.DataFrame({i:d})
		tmp_df = tmp_df[gv.game_vars['stock']['start']:gv.game_vars['stock']['end']].copy()
		return tmp_df

	# get overlay and indicator plot data
	ap_overlays = eval_overlays(df, gv)
	ap_indicators, ind_panel_manager, ind_panel_counter = eval_indicators(df, gv)

	# make array of plots from indicators
	for ind in ap_indicators:
		if ap_indicators[ind] is not None and ind_panel_counter < 9:
			if 'macd' in ind:
				if ind == 'macd':
					ind_panel_manager['macd'] = ind_panel_counter
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['macd'], color='k', width=1)
				if ind == 'macdsignal':
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['macd'], color='r', width=1)
				if ind == 'macdhist':
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['macd'], color='b', type='bar')
					ind_panel_counter += 1
				ap_arr.append(ap)

			elif 'stoch' in ind:
				if ind == 'stoch_slowk':
					ind_panel_manager['stoch'] = ind_panel_counter
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['stoch'], color='g', width=1)
				if ind == 'stoch_slowd':
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['stoch'], color='b', width=1)
					ind_panel_counter += 1
				ap_arr.append(ap)

			elif 'aroon' in ind:
				if ind == 'aroon_up':
					ind_panel_manager['aroon'] = ind_panel_counter
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['aroon'], color='g', width=1)
				if ind == 'aroon_down':
					tmp_df = _make_frame(ind, ap_indicators[ind])
					ap = mpf.make_addplot(tmp_df, panel=ind_panel_manager['aroon'], color='r', width=1)
					ind_panel_counter += 1
				ap_arr.append(ap)

			else:
				tmp_df = _make_frame(ind, ap_indicators[ind])
				ap = mpf.make_addplot(tmp_df, panel=ind_panel_counter, width=1)
				ap_arr.append(ap)
				ind_panel_counter += 1

	# make array of plots from overlays
	for ol in ap_overlays:
		if ap_overlays[ol] is not None:
			#print('for ol in ap_overlays. ol: {}, ap_overlays[ol]: {}'.format(ol, ap_overlays[ol]))
			if ol == 'psar':
				tmp_df = _make_frame(ol, ap_overlays[ol])
				ap = mpf.make_addplot(tmp_df, type='scatter', width=1)
			else:
				tmp_df = _make_frame(ol, ap_overlays[ol])
				ap = mpf.make_addplot(tmp_df, width=1)

			ap_arr.append(ap)

	return ap_arr


# calls class function to save data as csv
def save(gv, filename='./game_results.csv'):
	gv.save(filename=filename)


# runs if main program is given arg -u
def update_stocks():
	print('updating stocks...')

	ticker_df = download_tickers()
	ticker_list = format_tickers(ticker_df)
	stock_dict = get_stock_data(ticker_list)
	save_pkl(stock_dict, 'full_data')


# for update_stocks()
# gets all tickers. trimming is not done until format_tickers, which is probably a bad name
def download_tickers():
	print('download_tickers')

	headers = {
		"Host":"api.nasdaq.com",
		"Sec-Ch-Ua":'"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
		"Accept":"application/json, text/plain, */*",
		"Sec-Ch-Ua-Mobile":"?0",
		"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
		"Sec-Ch-Ua-Platform":'"Windows"',
		"Origin":"https://www.nasdaq.com",
		"Sec-Fetch-Site":"same-site",
		"Sec-Fetch-Mode":"cors",
		"Sec-Fetch-Dest":"empty",
		"Referer":"https://www.nasdaq.com/",
		"Accept-Encoding":"gzip, deflate",
		"Accept-Language":"en-US,en;q=0.9"
	}

	# get nasdaq
	nas = requests.get(url="https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=nasdaq&download=true", headers=headers)
	# get nyse
	nyse = requests.get(url="https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=nyse&download=true", headers=headers)
	# get amex
	amex = requests.get(url="https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=amex&download=true", headers=headers)

	# pull out k:v pairs
	nas_dict = nas.json()
	nyse_dict = nyse.json()
	amex_dict = amex.json()
	
	# convert to DataFrames
	nas_df = pd.DataFrame(nas_dict['data']['rows'], columns=nas_dict['data']['headers'])
	nyse_df = pd.DataFrame(nyse_dict['data']['rows'], columns=nyse_dict['data']['headers'])
	amex_df = pd.DataFrame(amex_dict['data']['rows'], columns=amex_dict['data']['headers'])

	# merge all DataFrames
	frames = [nas_df, nyse_df, amex_df]
	ticker_df = pd.concat(frames, ignore_index=True)
	
	# volume returns as String
	ticker_df['volume'] = pd.to_numeric(ticker_df['volume'])
	
	return ticker_df


# for update_stocks()
# takes DF generated by download_tickers, converts to a list of non-special symbols with specified av. vol, returns list of tickers
def format_tickers(ticker_df, vol_amt=100000):

	print('formatting tickers')

	# takes DF generated by download_tickers, prunes out tickers with vol below vol_amt, returns list of tickers
	def crop_list(ticker_df, vol_amt=vol_amt):
		symbol_list = list(ticker_df['symbol'][ticker_df['volume'] > vol_amt])
		return symbol_list
	
	ticker_list = []
	
	# presumably some of these symbols are meaningful and useful, but i'm sitting on an embarrassment of riches anyway
	# so just take them right out
	regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')

	# crop first, since cropping relies on DF
	tmp_ticker_list = crop_list(ticker_df)

	for sym in tmp_ticker_list:
		if(regex.search(sym) == None):
			if sym not in ticker_list:
				ticker_list.append(sym.rstrip())

	# just to make life a little easier when showing progress
	ticker_list.sort()
	
	return ticker_list


# for update_stocks()
# makes calls to yahoo_fin for each stock in list, rebuilds as dict, returns dict
def get_stock_data(stock_list):
	print('getting stock data')

	stock_array = []

	counter = 1
	for stock in stock_list[:]:
		print('working',counter,'of',len(stock_list[:]))
		try:
			sd = {}
			df = get_data(stock)
			
			sd['symbol'] = stock
			sd['df'] = df
			
			stock_array.append(sd)
		except:
			print('could not get',stock)
			continue
		counter += 1

	# rebuilds into more useful format
	stock_dict = {}

	print('rebuilding stock array as dict')

	for stock in stock_array:
		stock_dict[stock['symbol']] = {}
		stock_dict[stock['symbol']]['df'] = stock['df']

	return stock_dict


# for update_stocks()
# takes data, saves with name_date.pkl
def save_pkl(data, pkl_name):

	f = "./" + pkl_name + ".pkl"
	print('saving {}... '.format(f), end='')

	with open(f, 'wb+') as f:
		pkl.dump(data, f)
	
	print('done')





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~