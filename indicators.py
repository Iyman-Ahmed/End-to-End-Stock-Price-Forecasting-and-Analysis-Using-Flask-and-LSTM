from forecastingmodel import get_stock_data
import pandas as pd 

def sma(data):
    data = data.reset_index()
    data = data.sort_values(by = 'Date', ascending=True)
    data['Date'] = pd.to_datetime(data['Date'], format = '%d/%m/%Y')

    start_date = pd.to_datetime('01/05/2023', format='%d/%m/%Y').tz_localize('America/New_York')
    data = data[data['Date'] >= start_date]

    date = data['Date']
    sma_20 = data['SMA_20']
    sma_10 = data['SMA_10']
    sma_5 = data['SMA_5']
    close = data['Close']

    return date,sma_20,sma_10,sma_5
    
def ema(data):
    data = data.reset_index()
    data = data.sort_values(by = 'Date', ascending=True)
    data['Date'] = pd.to_datetime(data['Date'], format = '%d/%m/%Y')

    start_date = pd.to_datetime('01/05/2023', format='%d/%m/%Y').tz_localize('America/New_York')
    data = data[data['Date'] >= start_date]

    ema_20 = data['EMA_20']
    ema_10 = data['EMA_10']
    ema_5 = data['EMA_5']

    return ema_20,ema_10,ema_5    
