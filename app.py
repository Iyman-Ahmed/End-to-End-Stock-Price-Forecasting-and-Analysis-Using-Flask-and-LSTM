from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from forecastingmodel import get_stock_data, model_LSTM
from indicators import sma, ema  # Import the sma and ema functions

list_of_stocks = pd.read_csv("C:/Users/User/Downloads/nasdaq_screener_1728828417613.csv")

app = Flask(__name__)

# List of popular stock symbols (for demonstration)
popular_stocks = list_of_stocks['Symbol'].tolist()

@app.route('/')
def home():
    return render_template('index.html', stocks=popular_stocks)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    selected_stock = data['stock']
    
    # Get stock data and run the LSTM model
    stock_info = get_stock_data(selected_stock)
    actual_predictions, actual_values, predicted_today_price, index = model_LSTM(stock_info)
    
    predicted_today_price = round(predicted_today_price,2)
    # Calculate SMA and EMA values
    date_sma, sma_20, sma_10, sma_5 = sma(stock_info)
    ema_20, ema_10, ema_5 = ema(stock_info)

    # Reset and filter stock info for display
    stock_info = stock_info.reset_index()
    stock_info = stock_info.sort_values(by='Date', ascending=True)
    stock_info = stock_info[['Date', 'Open', 'High', 'Low', 'Close']]
    stock_info = stock_info.tail(10)

    # Convert stock info DataFrame to HTML table
    stock_info_html = stock_info.to_html(classes='table table-striped', index=False)

    # Create the LSTM model performance graph using Plotly
    fig_model = go.Figure()
    fig_model.add_trace(go.Scatter(x=index, y=actual_values, mode='lines', name='Actual Prices'))
    fig_model.add_trace(go.Scatter(x=index, y=actual_predictions, mode='lines', name='Predicted Prices', line=dict(color='red')))
    
    fig_model.update_layout(
        title=f'Model Performance on Live Data For {selected_stock} ',
        xaxis=dict(showticklabels=False),  # Hide date labels
        yaxis_title='Price',
        showlegend=True,
        width=800, height=400
    )

    # Create SMA graph using Plotly
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=date_sma, y=sma_20, mode='lines', name='SMA 20'))
    fig_sma.add_trace(go.Scatter(x=date_sma, y=sma_10, mode='lines', name='SMA 10'))
    fig_sma.add_trace(go.Scatter(x=date_sma, y=sma_5, mode='lines', name='SMA 5'))
    
    fig_sma.update_layout(
        title=f'{selected_stock} Simple Moving Averages (SMA)',
        yaxis_title='SMA Value',
        showlegend=True,
        width=800, height=400
    )

    # Create EMA graph using Plotly
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=date_sma, y=ema_20, mode='lines', name='EMA 20'))
    fig_ema.add_trace(go.Scatter(x=date_sma, y=ema_10, mode='lines', name='EMA 10'))
    fig_ema.add_trace(go.Scatter(x=date_sma, y=ema_5, mode='lines', name='EMA 5'))
    
    fig_ema.update_layout(
        title=f'{selected_stock} Exponential Moving Averages (EMA)',
        yaxis_title='EMA Value',
        showlegend=True,
        width=800, height=400
    )

    # Convert the Plotly graphs to JSON
    graph_model_json = pio.to_json(fig_model)
    graph_sma_json = pio.to_json(fig_sma)
    graph_ema_json = pio.to_json(fig_ema)

    return jsonify({
        "graph_model": graph_model_json,
        "graph_sma": graph_sma_json,
        "graph_ema": graph_ema_json,
        "predicted_today_price": predicted_today_price,
        "stock_info_html": stock_info_html,
        "selected_stock": selected_stock
    })

if __name__ == '__main__':
    app.run(debug=True)
