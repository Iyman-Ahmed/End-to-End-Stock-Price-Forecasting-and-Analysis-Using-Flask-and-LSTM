import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
from datetime import datetime


def get_stock_data(stock_name):
    stock = yf.Ticker(stock_name)
    dataset = stock.history(start = '2021-01-01')
    dataset['SMA_20'] = dataset['Close'].rolling(window=20).mean()
    dataset['SMA_10'] = dataset['Close'].rolling(window=10).mean()
    dataset['SMA_5'] = dataset['Close'].rolling(window=5).mean()
    dataset['EMA_20'] = dataset['Close'].ewm(span=20, adjust=False).mean()
    dataset['EMA_10'] = dataset['Close'].ewm(span=10, adjust=False).mean()
    dataset['EMA_5'] = dataset['Close'].ewm(span=5, adjust=False).mean()
    delta = dataset['Close'].diff(1)  # Calculate price changes
    gain = delta.where(delta > 0, 0)  # Keep only gains (positive changes)
    loss = -delta.where(delta < 0, 0)  # Keep only losses (negative changes)

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss

    # Calculate the RSI
    dataset['RSI_14'] = 100 - (100 / (1 + rs))

    dataset['STD_20'] = dataset['Close'].rolling(window=20).std()

    # Calculate the upper and lower Bollinger Bands
    dataset['Upper_BB'] = dataset['SMA_20'] + (2 * dataset['STD_20'])
    dataset['Lower_BB'] = dataset['SMA_20'] - (2 * dataset['STD_20'])
        
     # Calculate the 12-day and 26-day EMA
    dataset['EMA_12'] = dataset['Close'].ewm(span=12, adjust=False).mean()
    dataset['EMA_26'] = dataset['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    dataset['MACD'] = dataset['EMA_12'] - dataset['EMA_26']

    # Calculate the MACD s ignal line (9-day EMA of MACD)
    dataset['MACD_Signal'] = dataset['MACD'].ewm(span=9, adjust=False).mean()

    dataset.dropna(inplace = True)
    return dataset


# Visualise missing value after adding indicators in dataset
# import missingno as msno
# msno.matrix(data)
# plt.show()


def model_LSTM(data):
    data = data.dropna()
    index = data.index
    #Normalisation of Columns
    feature = ['Close', 'SMA_20', 'MACD', 'RSI_14']
    data = data[feature]

    data['Close_1'] = data['Close'].shift(1)
    data['Close_2'] = data['Close'].shift(2)
    data['Close_3'] = data['Close'].shift(3)
    
    data.dropna(inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_20', 'MACD', 'RSI_14', 'Close_1', 'Close_2', 'Close_3']])
    # Reshaping Data For LSTM
    X = []
    y = []
    time_steps = 60
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, :])
        y.append(scaled_data[i, 0])

    X,y = np.array(X), np.array(y)
    features = ['Close', 'SMA_20', 'MACD', 'RSI_14', 'Close_1', 'Close_2', 'Close_3']
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))
    # Defining Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=1, epochs=5)
    # Forecasting price
    predictions = model.predict(X_test)

    last_60_days_data = data.tail(60) 
    scaled_last_data = scaler.transform(last_60_days_data[['Close', 'SMA_20', 'MACD', 'RSI_14', 'Close_1', 'Close_2', 'Close_3']])

    X_today = []
    X_today.append(scaled_last_data)
    X_today = np.array(X_today)
    predicted_today = model.predict(X_today)
    predicted_today_price = scaler.inverse_transform([[predicted_today[0][0], 0, 0, 0, 0, 0, 0]])[0][0]
    dummy_array = np.zeros((predictions.shape[0], len(features)))  # Same number of features as scaler
    y_dummy_array = np.zeros((y_test.shape[0], len(features)))
    # Assign predictions to the first column
    dummy_array[:, 0] = predictions.flatten()  # Flatten to convert (n, 1) to (n,)
    y_dummy_array[:, 0] = y_test.flatten()
    # Inverse transform the predictions
    actual_predictions = scaler.inverse_transform(dummy_array)[:, 0]
    actual_values = scaler.inverse_transform(y_dummy_array)[:, 0]
    print(f"Predicted closing price for today: {predicted_today_price}")
    return actual_predictions, actual_values, predicted_today_price, index


def model_fb_pro(data):
    pro_data = data[['Close']].reset_index()
    pro_data.columns = ['ds','y']
    pro_data['ds'] = pd.to_datetime(pro_data['ds']).dt.tz_localize(None)
    model = Prophet(weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.05)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(pro_data)
    future = model.make_future_dataframe(periods = 30)
    forecast = model.predict(future )    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'Forecast for Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

    # Optional: Plot the forecast components
    model.plot_components(forecast)
    plt.show()

def lstm_pre_grapgh(actual_prices,predictions):

    dates = np.arange(len(actual_prices))
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(dates, predictions, label='Predicted Prices', color='orange')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


def model_arima(data):
    data = data[['Close']]
    # plot_acf(data)
    # plt.title('ACF Graph')
    # plt.show()
    # plot_pacf(data)
    # plt.title('PACF Graph')
    # plt.show()
    
    p = 2
    d = 1
    q = 0

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()

    # Step 6: Make Predictions
    forecast = model_fit.forecast(steps=30)  # Forecast for the next 30 days
    print(forecast)

    last_date = data.index[-1]  # Get the last date from your historical data

# Generate a new index for the forecast starting from the day after the last historical date
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast))

    # Plot the historical prices and forecasted prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Prices')
    plt.plot(forecast_dates, forecast, label='Forecasted Prices', color='red')
    plt.title('Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
