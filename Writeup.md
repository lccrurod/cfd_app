## Project Overview

The objective of the project is to build a machine leaning driven web application to aid in the investing process in CFD instruments.

A contract for differences (CFD) is finalcial derivative, which means its value depends on the subjacent asset, the value of the contract can be determined by the difference between the current and the time of contract value of the stock, thus the contract can have positive or negative value and yield benefitial for either the buyer or the seller respectively.

The main idea is to make predictions based on stock data consulted by the [Alpha Vantage API](https://www.alphavantage.co/documentation/) from a web app, and these aid the decision making of a potencial CFD investor.

## Problem Statement

For this type of instrument, knowing in advance whether the stock value will rise o fall would translate into knowing which position to take in order to make money out of the invesment. Also it is necesary to know when to finish the contract or close the position in it to make the maximun profit, for that, it is necesary to know the upper and lower bounds of the stock value for the future.

Based on that, there are two challenges, based on historical data from a stock both predict, in a given time, if the value will go up or down compared to the current value and what are going to be the most extreme lower and upper values.

Thee proposed web app lets the user choose between four selected stocks, the most traded ones at the moment of development, and return it's prediction based on last business day data available to whether stock value will rise or fall in the following three months and the maximun and minimun value for the period.

To develop such app will be necesary to fit both a classification and a regression algorith to the historical data, and make those trained models available to a web application, The app itself must call the data API to get the current data of the stock, trasform it to the features used for the training and retrieved the prediction for bot discussed challenges.

## Metrics

### Classification: 
As the result from this process may be considered when taking a position in the contract, since both positions could yield a benefit(or a loss), predictions for both classes should be good and because these classes are well balanced, the `accuracy_score` is a good metric to evaluate the performance. 


### Regression: 
The result of this process may aid when calculating maximun benefits, for example from the perspective of a buyer when the stock value reaches the maximum for the period the position should be closed so the maximun benefit is withdraw, since the `r2_score` metric main purpose is the measure of predict future outcomes performance, this is the ideal metric to evaluate this task.


## Data Exploration

The retrieved data from the API request is a dictionary that contains two big main items.

### 1. The Request Metadata 
Having the parameters used in the request plus time and timezone when the request was realized.

### 2. The Time Series
Transform into pandas dataframe is made of the following columns about the stock:

#### 2.0 index
Date to which the information corresponds.

#### 2.1 open
Opening price for the date.

#### 2.2 high
Maximun price for the date

#### 2.3 low
Minimum price for the date

#### 2.4 close
Closing price for the date, this is the main variable as its common to model the stock return, price and volatility based on the closing price for a period.

#### 2.5 adjusted close
Adjusted closing price for the date, for more information [investopedia](https://www.investopedia.com/terms/a/adjusted_closing_price.asp)

#### 2.6 volume
Number of transactions for the date.

The data comes decreaseanly ordered by index, without missing values and only business days dates.

## Data Visualization

![Tesla Monthly](https://github.com/lccrurod/cfd_app/blob/main/tesla_monthly_plot.png)

![Tesla Weekly](https://github.com/lccrurod/cfd_app/blob/main/tesla_weekly_plot.png)

Plotted visualizations of the monthly and weekly time series for the Tesla's symbol.

## Data Preprocessing

For the data retrieved API request the columns names are adjusted to remove the digit and space after, all fields are converted to numeric and based on the index date are added the columns `year`, `month` and `week`.

The main function for this task is 

```
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
    
    # add individual date related features for merging different frequency data
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
```

The initial features considered were the summarized mean values acording to the last n = 100 data points for the day series, weekly series, and n = 60 for the monthly series, this allows to cover a range from ~3 months to 5 years giving balance information for short, mid and long term data.

For that process was coded the following class.

```
class LoadStockFeatures(BaseEstimator, TransformerMixin):
    def load_freq_data(self, stock_symbol, freq):
        """
        Load stock_symbol data for the defined frequency.
        """
        engine = create_engine('sqlite:///'+'stock_price.db')
        df = pd.read_sql_table(stock_symbol + '_' + freq, engine)
        df['index'] = pd.to_datetime(df['index'])
        return df
    
    def build_summarized_features(self, df, n_points):
        """
        Get the mean of the last n_points for the value columns
        """
        
        val_cols = ['open',
                    'high',
                    'low',
                    'close',
                    'adjusted close',
                    'volume']
        
        # initialize the mean columns
        for col in val_cols:
            df['mean_'+col] = -999
            
            # scan the dataframe backwards in time
            for i in range(df.shape[0]-n_points):
                # assign the mean of the last n observations to the relative entry
                df.loc[i, 'mean_'+col] = df.loc[i:i+n_points, col].mean()
            
        return df
    
    def build_obj_vals(self, df, f_months):
        # initialize columns
        df['obj_rise'] = -999
        df['obj_max'] = -999
        df['obj_min'] = -999
        
        # scan the dataframe backwards in time
        for i in range(df.shape[0]):
            
            # assing start and end date
            start_date = df.iloc[i,0]
            end_date = start_date + pd.DateOffset(months=f_months)
            
            # check if there are entries at or after the end date
            if df['index'].max() < end_date:
                continue
            # get data from start to end date
            df_3_month = df[(df['index']> start_date) & (df['index']<= end_date)]
            
            # assing realization of the minimun and maximun values
            df.loc[i, 'obj_max'] = df_3_month['close'].max()
            df.loc[i, 'obj_min'] = df_3_month['close'].min()
            
            # identify wheter the stock value increase compared to starting price
            if df_3_month.tail(1)['close'].values[0] > df.loc[i, 'close']:
                df.loc[i, 'obj_rise'] = 1
            else:
                df.loc[i, 'obj_rise'] = 0
        
        return df
    
    def decode_nulls(self, df):
        """
        Replace initialized only values in the data with np.nan.
        """
        decode_null = lambda x: x.replace(-999, np.nan)
        df = df.apply(decode_null, axis = 1)
        return df
    
    def fit(self):
        return self
    
    def transform(self, stock_symbol, n_day = 100, n_week = 100, n_month = 60, f_months = 3):
        """
        Run complete feature and objetive values building process for stock_symbol.
        """
        df_d = self.load_freq_data(stock_symbol, 'daily')
        df_w = self.load_freq_data(stock_symbol, 'weekly')
        df_m = self.load_freq_data(stock_symbol, 'monthly')
        print('Loaded time series for the symbol...')
        
        df_d = self.build_summarized_features(df_d, n_day)
        df_w = self.build_summarized_features(df_w, n_week)
        df_m = self.build_summarized_features(df_m, n_month)
        print('Builded sumarized features for the symbol...')
        
        df_d = self.build_obj_vals(df_d, f_months)
        print('Builded objetive variables for the symbol...')
        
        df = df_w.merge(df_m,
                         'left',
                         left_on = ['year', 'month'],
                         right_on = ['year', 'month'],
                         suffixes = ('_weekly', '_monthly'))
        df = df_d.merge(df,
                        'left',
                        left_on = ['year', 'week'],
                        right_on = ['year', 'week_weekly'],
                        suffixes = ('','_weekly'))
        print('Merged all relevant data...')
        df = df[df.columns[(df.columns.str.contains('mean')) | (df.columns.str.contains('obj'))]]
        df = self.decode_nulls(df)
        print('Decoded null entries...')
        
        df = df.dropna()
        print('Dropped null entries...')
        print(' Succesfully loaded symbol features!')
        return df
```

Take notice that this is an `estimator` object, but it isn't used inside a `Pipeline` since separating data points before this feature building process would completly destroy the proposed features.


## Implementation

Since two estimators needed to be trained and evaluated over the same data set, it made sense to execute that process simultaneously, with that in mind the `ClassifierRegressorCombo` object was coded.

```
class ClassifierRegressorCombo():
    """
    Combination of classifier and regressor object to predict over the same features.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    def __init__(self, df, clf = RandomForestClassifier(), reg = RandomForestRegressor(), clf_acc = 0, r2_max = 0, r2_min = 0):
        self.df = df
        self.clf = clf
        self.reg = reg
        self.clf_acc = clf_acc
        self.r2_max = r2_max
        self.r2_min = r2_min
    
    def set_df(self, df):
        self.df = df
    
    def fit(self):
        """
        Run fitting process for classifier and regressor objects.
        """
        X = self.df[self.df.columns[self.df.columns.str.contains('mean')]]
        clf_y = self.df['obj_rise']
        reg_y = self.df[['obj_max','obj_min']]
        self.clf.fit(X, clf_y)
        self.reg.fit(X, reg_y)
        
    def predict(self):
        """
        Run predict method for classifier and regressor objects.
        """
        X = self.df[self.df.columns[self.df.columns.str.contains('mean')]]
        clf_pred = self.clf.predict(X)
        reg_pred = self.reg.predict(X)
        return clf_pred, reg_pred
    
    def clf_report(self, clf_true, clf_pred):
        """
        Get metrics for the classifier.
        """
        print(classification_report(clf_true, clf_pred))
        self.clf_acc = accuracy_score(clf_true, clf_pred)
        
    def reg_report(self, reg_true, reg_pred):
        """
        Get metrics for the regressor.
        """
        print(" R2 for the maximum value regression: {}".format(r2_score(reg_true['obj_max'], reg_pred[:, 0])))
        self.r2_max = r2_score(reg_true['obj_max'], reg_pred[:, 0])
        print(" R2 for the minimun value regression: {}".format(r2_score(reg_true['obj_min'], reg_pred[:, 1])))     
        self.r2_min = r2_score(reg_true['obj_min'], reg_pred[:, 1])
    def full_test(self):
        """
        Train, predict and get metrics for the classifier and regressor.
        """
        df = self.df
        # split data into test and testing sets
        df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
        print("Splited train and test sets...")
        
        # use training data for fitting method
        self.set_df(df_train)
        self.fit()
        print("Trained models on train set...")
        
        # use testing data for predict method
        self.set_df(df_test)        
        clf_pred, reg_pred = self.predict()
        
        # display and store evaluation metrics
        print("Predicted on test set...")
        print("Classification Report: {}".format(self.clf.__class__.__name__))
        self.clf_report(df_test['obj_rise'], clf_pred)
        
        try:
            print("Regresion Report: {}".format(self.reg.estimator.__class__.__name__))
        except:
            print("Regresion Report: {}".format(self.reg.__class__.__name__))
            
        self.reg_report(df_test[['obj_max','obj_min']], reg_pred)
        return None
```

This made it more agile to further test and evaluate models for both of the proposed challenges.

## Refinement

For the refinemnt of the predictions made two approaches were taken:

### Extend the feature set

Three* aditional features were added

#### 1 Average of the last 100 daily returns of the stock.

This returns are the main modeling tool in the [modern portafolio theory.](https://www.jstor.org/stable/2975974)

#### 2 Percentage of the last 100 days that the stock is below the current value.

As the classification can be interpreted as predicting a bernoulli random variable, this feature would correspond to the maximum likelihood estimator for the mean of that variable, given the sample of the last 100 days.

#### 3 Polynomial Features.

This type of transformations would allows to include interactions within features and maybe allow the models to grasp better the relevant patterns.

For detail info [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=poly#sklearn.preprocessing.PolynomialFeatures)

*more than 3 features since polynomial features includes at least n_features-1.

### GridSearCV

Performed grid search fitting for both the Classifier and Regressor estimators across all 4 stock symbols, but because the regressor was performing remarkably out of the box, poly features wasn't added to its pipeline. The used code was the following.

```
def build_models():
    """
    Ensemble full regressor and classifier pipelines.
    """
    clf_param_dict = {'clf__max_depth':[2, 3, 5],
                      'polyfeatures__interaction_only':(True, False)}
    reg_param_dict = {'n_estimators':[75,100,125],
                      'max_depth':[3, 6, None]}
    
    gbc = Pipeline([('polyfeatures',PolynomialFeatures()),
                    ('clf', GradientBoostingClassifier())])
                    
    rfr = RandomForestRegressor() 
    
    clf = GridSearchCV(gbc, param_grid = clf_param_dict)
    reg = GridSearchCV(rfr, param_grid = reg_param_dict)
    
    return clf, reg

def save_models(clf, reg, symbol):
    """
    Save models for symbol stock value classification and regression.
    """
    filename = 'models/'+symbol+'_clf.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    
    filename = 'models/'+symbol+'_reg.pkl'
    pickle.dump(reg, open(filename, 'wb'))
    
    
def save_metrics(metric_dict):
    engine = create_engine('sqlite:///'+'stock_price.db')
    df = pd.DataFrame(metric_dict)
    df.to_sql('model_metrics', engine, index = False, if_exists= 'replace')
    
def main_models():
    
    symbols = ['AAPL','FCEL','BAC','M']
    
    # initialize metric dictionary
    metric_dict = {'symbol':symbols,
                   'classification accuracy':[],
                   'r2 score - max':[],
                   'r2 score - min':[]}
    
    for symbol in symbols:
        print("\n -------- Started Process for {} -------- \n".format(symbol))
        start_time = time.time()
        # instantiate feature extractor object
        lsef = LoadStockFeaturesExtended()
        # get extended feature set for the symbol
        df = lsef.transform(symbol)
        
        # perform train test test evaluation
        df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
        print('Train a test sets splitted')
        print('Building models...')
        clf, reg = build_models()
        
        X_train = df_train[df_train.columns[df_train.columns.str.contains('mean')]]
        y_clf_train = df_train['obj_rise']
        y_reg_train = df_train[['obj_max','obj_min']]
        
        print('Fitting Classifier...')
        clf.fit(X_train, y_clf_train)
        print('Fitting Regressor...')
        reg.fit(X_train, y_reg_train)
    
        X_test = df_test[df_test.columns[df_test.columns.str.contains('mean')]]
        print('Predicting on testing set...')
        y_clf_pred = clf.predict(X_test)
        y_reg_pred = reg.predict(X_test)
        
        # store metrics
        metric_dict['classification accuracy'].append(accuracy_score(df_test['obj_rise'], y_clf_pred))
        metric_dict['r2 score - max'].append(r2_score(df_test['obj_max'], y_reg_pred[:, 0]))
        metric_dict['r2 score - min'].append(r2_score(df_test['obj_min'], y_reg_pred[:, 1]))
        
        print('Saving models...')
        save_models(clf, reg, symbol)
        print(" elapsed time: {}".format(time.time()- start_time))
    print('Saving overall metrics...')
    save_metrics(metric_dict)
    print('  Training process completed!')
```


## Model Evaluation and Validation

To chose the initial models was evaluated the out of the box performance for several models here are the results.

![Classifiers Results](https://github.com/lccrurod/cfd_app/blob/main/clf_out_of_the_box.png)

![Regressors Results](https://github.com/lccrurod/cfd_app/blob/main/reg_out_of_the_box.png)

Based on those results and considering that for each stock it's necesary a pair of estimators, were chosen the best on average across all stocks.

 GradientBoostingClassifier	`0.518795`

 RandomForestRegressor `0.997638`

These were then optimized as mentioned in the refinement section using GridSearchCV.

| symbol | clf__max_depth | polyfeatures__interaction_only |
|---|---|---|
| AAPL | 2 | True |
| BAC | 5 | False |
| FCEL | 2 | False |
| M | 2 | True |

Average accuracy_score across all stocks `0.519975`

| symbol | max_depth | n_estimators |
|---|---|---|
| AAPL | None | 125 |
| BAC | None | 125 |
| FCEL | None | 100 |
| M | None | 100 |

Average r2_score across all stocks `0.999212`

## Justification

It is evident from the results that the classification process didn't improve significantly from the optimization and extedend features, nonetheless the accuracy score for this is above 50% meaning that if the success of an invesment depended only in the accuracy of this classification, repeating that invesment on average would yield benefits, obviously ignoring comissions an many other aspects. The results for the prediction of the minimun and maximun prediction were remarkable, and could yield to investing estrategies on its own.


## Reflection

As for the entire project, personally the development of the web app prove very challenging, but the most rewarding once everything is running, the fact that bootstrap documentation is splited in 4.0 and 5.0, and in the course videos references 3.0 that is not available didn't make easier, finally it was necesary to take a more pragmatical approach and prioritize function over shape and make the visualization for example by javascrip rather than by passing it by python plotly.

Also would be very intersting to actually use this app for real investing, the CFDs are one of the most riskier financial derivatives, both for volatility and the hability to take leveraged positions, but aided by this could maximize the capitalization posibilities with this instrument.


## Improvement

As an opportunity for improvement, maybe defining a maximal relative loss on the buyer position during a certain period could be a more reasonable objective variable to try to predict, or not defining if the stock value rises or falls according to a specific day but maybe a range, like from the 3th to the 4th month. In other words, the classification challenge could be rephrased for a more realistic/harder to code approach that could prove easier to predict.

Also for the web app would be nice to include proyection of loss or earnings according to a position and the leverage.
