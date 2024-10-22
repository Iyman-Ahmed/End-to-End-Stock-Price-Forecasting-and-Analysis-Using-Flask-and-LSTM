# LSTM-Powered Flask Application for Stock Market Analysis and Forecasting

This project is a Flask-based web application that provides real-time stock market analysis and forecasting using a Long Short-Term Memory (LSTM) model. The application allows users to select stocks, view historical price data, analyze trends using technical indicators, and forecast future prices based on historical data.

## Features

- **Stock Price Forecasting**: Uses an LSTM model to predict future stock prices.
- **Technical Indicators**: Visualizes key indicators such as Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
- **Real-Time Data**: Retrieves stock data from the yFinance API to perform real-time analysis.
- **Interactive Dashboard**: Provides an intuitive interface for users to interact with stock data and view predictions.

## Project Structure

```bash
├── forecastingmodel.py       # Contains the LSTM model and stock data retrieval functions
├── indicators.py             # Functions for calculating SMA and EMA
├── templates/
│   └── index.html            # Frontend HTML template for the dashboard
├── static/
│   └── style.css             # Custom CSS for styling the frontend
├── app.py                    # Flask server and routes for backend logic
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies required to run the project


#File Descriptions
##app.py:

This file is the main entry point for the Flask application.
It handles routing and processes user inputs from the web interface.
It uses functions from forecastingmodel.py and indicators.py to perform stock analysis, prediction, and visualization.
The results (including graphs and predictions) are displayed on the webpage using Plotly.

##forecastingmodel.py:

This file contains the logic for the Long Short-Term Memory (LSTM) model, which is used to forecast stock prices.
It also includes the function to fetch historical stock data using the yFinance API.
The LSTM model processes the stock data and outputs the predicted prices for the selected stock.

###Functions:

get_stock_data(stock_name):

Retrieves historical stock data from Yahoo Finance (using the yfinance API).
Adds several technical indicators such as:
Simple Moving Averages (SMA) for 5, 10, and 20 days.
Exponential Moving Averages (EMA) for 5, 10, 20, 12, and 26 days.
Relative Strength Index (RSI) for a 14-day window.
Bollinger Bands (Upper/Lower bands based on SMA).
Moving Average Convergence Divergence (MACD) and its signal line.
Returns a pandas DataFrame with the calculated indicators.
model_LSTM(data):

Builds and trains a Long Short-Term Memory (LSTM) model to predict future stock prices.
Prepares the data by normalizing it and creating lag features (Close_1, Close_2, Close_3).
Splits the data into training and testing sets.
Defines the LSTM model architecture using Keras, with two LSTM layers and dropout layers for regularization.
Trains the model and predicts future stock prices.
Returns predicted prices, actual values, predicted today's price, and the index for visualization.
model_fb_pro(data):

Uses Facebook's Prophet model for time-series forecasting of stock prices.
Trains the Prophet model and forecasts stock prices for the next 30 days.
Plots the forecast and its components.
lstm_pre_grapgh(actual_prices, predictions):

Visualizes the actual and predicted stock prices on a graph.
Plots the comparison between actual closing prices and LSTM-predicted prices.
model_arima(data):

Builds and fits an ARIMA model to forecast stock prices.
Uses Auto-Correlation Function (ACF) and Partial Auto-Correlation Function (PACF) to determine the order of the ARIMA model.
Forecasts prices for the next 30 days and plots the forecasted prices along with the historical prices.
Technical Indicators Implemented:
SMA (Simple Moving Average): Calculated for 5, 10, and 20 days.
EMA (Exponential Moving Average): Calculated for 5, 10, 12, 20, and 26 days.
RSI (Relative Strength Index): Measures the strength of recent price changes.
Bollinger Bands: Helps assess volatility by plotting standard deviations above and below SMA.
MACD (Moving Average Convergence Divergence): A trend-following momentum indicator.
Machine Learning Models:
LSTM: Used to predict future stock prices based on historical data and technical indicators.
Prophet: Facebook's time-series forecasting tool, useful for daily and weekly trends.
ARIMA: A statistical model for time-series forecasting that focuses on auto-regressive properties.


##indicators.py:

This file contains functions for calculating technical indicators such as Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
These indicators are used for analyzing stock trends, and the calculated values are displayed in graphs on the dashboard.

#Dependencies
The following libraries are required and can be installed via requirements.txt:

Flask: For running the web application.
pandas: For data manipulation and handling stock data.
numpy: For numerical operations, particularly in data preparation for the LSTM model.
matplotlib: For basic plotting, though primarily replaced by Plotly in this app.
seaborn: For advanced data visualization.
prophet: For forecasting using Prophet (Facebook's time series forecasting model).
statsmodels: For time-series models like ARIMA.
tensorflow: For building and training the LSTM model.
scikit-learn: For machine learning algorithms, particularly Random Forest and other ML models.
xgboost: For Gradient Boosting using XGBoost.
lightgbm: For Gradient Boosting using LightGBM.
yfinance: For retrieving stock data from Yahoo Finance.
missingno: For visualizing missing data in the dataset.
plotly: For creating interactive and dynamic graphs for stock prices, SMA, and EMA.

#Future Enhancements
Use LLM to analyse News data for predicting unexpected movements.
Implement additional technical indicators (e.g., Bollinger Bands, MACD).
Add user authentication for personalized stock analysis dashboards.
Optimize the LSTM model through hyperparameter tuning for improved accuracy.