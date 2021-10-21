import pandas as pd
import numpy as np
import requests
import re
from sqlalchemy import create_engine
import time
import pickle
import datetime as dt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin



def load_series(symbol, frequency, outputsize = 'compact'):
    """
    Load time serie from the stock symbol with the defined frequency and outputsize.
    
    IN:
    symbol -- stock's symbol to get the data.
    frequency -- either dayly, weekly or monthly data to be retrieved.
    outputsize -- by default compact for last 100 data points in the series or 
                  full for the complete 20+ years of historical data.
    
    OUT: pandas dataframe with the following columns:
    open -- opening price on the period.
    high -- maximun price during the period.
    low -- minimum price during the period.
    close -- closing price on the period.
    adjuste close -- adjusted closing price on the period.
    volume -- number of negotiations during the period.
    """
    
    # function dictionary for different time series in the API
    freq_funct_dict = {'daily':'TIME_SERIES_DAILY_ADJUSTED',
                       'weekly':'TIME_SERIES_WEEKLY_ADJUSTED',
                       'monthly':'TIME_SERIES_MONTHLY_ADJUSTED'}
    
    # define the url to make the request
    url = 'https://www.alphavantage.co/query?function='+freq_funct_dict[frequency]+'&symbol='+symbol \
        +'&outputsize='+outputsize+'&apikey=F0UT6370FXK949PA'
    
    r = requests.get(url)
    data = r.json()
    
    # build dataframe from data dictionary
    df = pd.DataFrame.from_dict(data = data[list(data.keys())[1]], orient='index')
    
    # adjust columns types and names
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    rem_dig = lambda x : re.sub('[1-9]. ', '', x)
    df.columns = [rem_dig(x) for x in df.columns]
    
    df = df.reset_index()
    df['index'] = pd.to_datetime(df['index'])
    
    df['year'] = df['index'].dt.year
    df['month'] = df['index'].dt.month
    df['week'] = df['index'].dt.isocalendar().week
    
    df = df.set_index('index')
    
    out_cols = ['open',
                'high',
                'low',
                'close',
                'adjusted close',
                'volume',
                'year',
                'month',
                'week']
    return df[out_cols]

def save_pred_data(symbol):
    """
    Save the stock data in the defined database
    """
    engine = create_engine('sqlite:///../stock_price.db')

    day_request = dt.date.today().strftime('%Y%m%d')

    for freq in ['daily', 'weekly', 'monthly']:
        try:
            df = pd.read_sql_table(stock_symbol + '_' + freq+'_to_predict', engine)
            df = df[df['day_request']==day_request]
            if df.shape[0] > 0:
                print('.. Data already have been requested {}..'.format(day_request))
                return None
        except:
            df = load_series(symbol, freq, outputsize = 'compact')
            df['day_request']=day_request
            df.to_sql(symbol+'_'+freq+'_to_predict', engine, if_exists = 'replace')

class LoadPredFeatures():
    def load_freq_data(self, stock_symbol, freq):
        engine = create_engine('sqlite:///../stock_price.db')
        df = pd.read_sql_table(stock_symbol + '_' + freq+'_to_predict', engine)
        df['index'] = pd.to_datetime(df['index'])
        return df
    
    def get_features(self, df, n_points, extended = False):
        df = df.head(n_points)
        cols = ['open','high','low','close','adjusted close','volume']
        df_ = df[cols].apply(np.mean, axis = 0)
        df_ = pd.DataFrame(df_).transpose()
        df_.columns = ['mean_'+col for col in cols]
        df_['m_key'] = 1
        if extended:
            df_['mean_return'] =np.mean(np.log(np.divide(np.array(df.loc[0:n_points-2, 'close']), 
                                                         np.array(df.loc[1:n_points-1, 'close']))))

            df_['mean_days_below'] = np.mean(np.array(df.loc[:, 'close'])<df.loc[0, 'close'])
                    
        return df_
    
    def transform(self, stock_symbol, n_day = 100, n_week = 100, n_month = 60, f_months = 3):
        df_d = self.load_freq_data(stock_symbol, 'daily')
        df_w = self.load_freq_data(stock_symbol, 'weekly')
        df_m = self.load_freq_data(stock_symbol, 'monthly')
        print('Loaded time series for the symbol...')
        
        df_d = self.get_features(df_d, n_day, extended = True)
        df_w = self.get_features(df_w, n_week)
        df_m = self.get_features(df_m, n_month)
        print('Builded features for the symbol...')
        
        #df_d = self.build_obj_vals(df_d, f_months)
        #print('Builded objetive variables for the symbol...')
        
        df = df_w.merge(df_m,
                         'left',
                         left_on = 'm_key',
                         right_on = 'm_key',
                         suffixes = ('_weekly', '_monthly'))
        df = df_d.merge(df,
                        'left',
                        left_on = 'm_key',
                        right_on = 'm_key',
                        suffixes = ('','_weekly'))
        print('Merged Features...')
        df = df[df.columns[df.columns.str.contains('mean')]]
        print(' Succesfully loaded symbol features!')
        return df

def load_models(symbol):
    clf = pickle.load(open('../models/'+symbol+'_clf.pkl', 'rb'))
    reg = pickle.load(open('../models/'+symbol+'_reg.pkl', 'rb'))
    return clf, reg

if __name__ == '__main__':
    print('loaded modules')