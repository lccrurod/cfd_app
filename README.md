# cfd_app

## About the project

This repository is part of the final project for Data Scientist Nanodegree program from udacity.

The objective of the project is to build a machine leaning driven web application to aid in the investing process in CFD instruments.

A contract for differences (CFD) is finalcial derivative, which means its value depends on the subjacent asset, the value of the contract can be determined by the difference between the current and the time of contract value of the stock, thus the contract can have positive or negative value and yield benefitial for either the buyer or the seller respectively.

For this type of instrument, knowing in advance whether the stock value will rise o fall would translate into knowing which position to take in order to make money out of the invesment. Also it is necesary to know when to finish the contract or close the position in it to make the maximun profit, for that it is necesary to know the upper and lower bounds of the stock value for the future.

With all that said, the proposed web app lets the user choose between four selected stocks and return it's prediction based on last business day data to whether stock value will raise or fall in the following three months and the maximun and minimun value for the period.

The data both for modeling and web app prediction was extracted using Alpha Vantages API.

## Files

### app
#### templates/home.html
Main page to be show in the web app.

#### templates/go.html
Page to be show when the user choses a stock.

#### aux_models.py
Module that contains auxiliar utilities to be used in the app.

#### run.py
Backend code for the web app.

### models
#### [stock symbol]_clf.pkl
Pickle stored ML model for prediction on rise or fall of the stock value.

#### [stock symbol]_reg.pkl
Pickle stored ML model for prediction about the maximun and minimum value of the stock.

### Data Processing.ipynb
Notebook with detail analysis of the API used an necesary transformations to the data for modeling process.

### CFD Modeling.ipynb
Notebook with extedend detail of the feature construction, model selection, feature engineering, model optimization and models saving.

### stock_price.db
Database to store requested data en model performance metrics.
## Libraries Used

This are available in the **requirements.txt** file

## Execution
The data preparation and modeling process can only be perform by executing the jupyter notebooks (not necesary as all the trained models are already stored in models folder and data results stored in .db file)

### 1. Data Processing.ipynb

### 2. CFD Modeling.ipynb

### 3. The web app: for executing the the web app perform the following steps:

From the app directory run

>`set FLASK_APP=run`
>
>`flask run`

### The web app should now be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/")

## The web app

![First screenshot](https://github.com/lccrurod/cfd_app/blob/main/web_app_ss1.PNG)

![Second screenshot](https://github.com/lccrurod/cfd_app/blob/main/web_app_ss2.PNG)


## Acknowlegments

Special thanks to the udacity team, I suppose this depends on the point of view but for me, working with these projects in the [Data Scientist Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025, "Udacity") feels like learning superpowers!

Also was life saving this [documentation](https://plotly.com/javascript/) from the plotly team at the moment of generating that visual in the web app.

This [paper](http://worldcomp-proceedings.com/proc/p2012/ICA4753.pdf) published by members of the Institute of Artificial Intelligence from the University of Georgia was part of the motivation for chosing this kind of project and served as a guide for the features to be used later in the modeling process.

The ease of use and setup for the [Alpha Vantage API](https://www.alphavantage.co/documentation/) made the data processing step pass like a breeze!

Finally for setting my local CMD for running the flask app and succesfully excecute the web app I refered to the  [quickstart documentation](https://flask.palletsprojects.com/en/2.0.x/quickstart/, "flask-doc") from the Flask team.


