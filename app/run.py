from flask import Flask
from flask import render_template, url_for

import pandas as pd
from sqlalchemy import create_engine

from aux_models import save_pred_data, LoadPredFeatures, load_models

import plotly.graph_objs as GO
import plotly, json

engine = create_engine('sqlite:///../stock_price.db')
metrics_df = pd.read_sql_table('model_metrics', engine)

stock_dict = {'BAC':'Bank of America Corporation.',
			  'M':"Macy's, Inc.",
			  'FCEL':'FuelCell Energy, Inc.',
			  'AAPL':'Apple Inc.'}

app = Flask(__name__)

@app.route('/')
@app.route('/index')

def index():

	# data for displaying evaluation metrics for the current models.
	BAC_clf_acc = round(list(metrics_df[metrics_df['symbol'] == 'BAC']['classification accuracy'])[0]*100,2)
	BAC_reg_r2_max = round(list(metrics_df[metrics_df['symbol'] == 'BAC']['r2 score - max'])[0]*100,2)
	BAC_reg_r2_min = round(list(metrics_df[metrics_df['symbol'] == 'BAC']['r2 score - min'])[0]*100,2)

	M_clf_acc = round(list(metrics_df[metrics_df['symbol'] == 'M']['classification accuracy'])[0]*100,2)
	M_reg_r2_max = round(list(metrics_df[metrics_df['symbol'] == 'M']['r2 score - max'])[0]*100,2)
	M_reg_r2_min = round(list(metrics_df[metrics_df['symbol'] == 'M']['r2 score - min'])[0]*100,2)
	
	FCEL_clf_acc = round(list(metrics_df[metrics_df['symbol'] == 'FCEL']['classification accuracy'])[0]*100,2)
	FCEL_reg_r2_max = round(list(metrics_df[metrics_df['symbol'] == 'FCEL']['r2 score - max'])[0]*100,2)
	FCEL_reg_r2_min = round(list(metrics_df[metrics_df['symbol'] == 'FCEL']['r2 score - min'])[0]*100,2)
	
	AAPL_clf_acc = round(list(metrics_df[metrics_df['symbol'] == 'AAPL']['classification accuracy'])[0]*100,2)
	AAPL_reg_r2_max = round(list(metrics_df[metrics_df['symbol'] == 'AAPL']['r2 score - max'])[0]*100,2)
	AAPL_reg_r2_min = round(list(metrics_df[metrics_df['symbol'] == 'AAPL']['r2 score - min'])[0]*100,2)
	return render_template('home.html',
							BAC_clf_acc=BAC_clf_acc, BAC_reg_r2_max=BAC_reg_r2_max, BAC_reg_r2_min=BAC_reg_r2_min,
							M_clf_acc=M_clf_acc, M_reg_r2_max=M_reg_r2_max, M_reg_r2_min=M_reg_r2_min,
							FCEL_clf_acc=FCEL_clf_acc, FCEL_reg_r2_max=FCEL_reg_r2_max, FCEL_reg_r2_min=FCEL_reg_r2_min,
							AAPL_clf_acc=AAPL_clf_acc, AAPL_reg_r2_max=AAPL_reg_r2_max, AAPL_reg_r2_min=AAPL_reg_r2_min)

@app.route('/go/<string:symbol>')

def go(symbol):
	
	# request data for the symbol stock
	save_pred_data(symbol)

	# transform data into features for prediction
	lpf = LoadPredFeatures()
	df = lpf.transform(symbol)

	# load models for the symbol
	clf, reg = load_models(symbol)

	# prediction
	y_clf = clf.predict(df)
	if y_clf[0] == 1:
		clf_label = 'will go up!'
	else:
		clf_label = 'will go down!'

	y_reg = reg.predict(df)

	# data for visualization 
	df = lpf.load_freq_data(symbol, 'daily')
	
	return render_template('go.html',
							stock_name = stock_dict[symbol], 
							clf_label = clf_label,
							max_value = round(y_reg[0, 0],4),
							min_value = round(y_reg[0, 1],4),
							x = list(df['index'].astype('str')), y = list(df['close']))

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()