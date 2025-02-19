import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import talib

# Load Pretrained LSTM Model
model = load_model("lstm_model.h5")  

# Function to fetch live stock data
def get_live_data(ticker, days=90):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to make real-time predictions
def predict_price(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']])
    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    predicted_price = model.predict(last_60_days)
    return scaler.inverse_transform(predicted_price)[0][0]

# Function to generate trading signals
def get_trading_signal(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    last_price = df['Close'].iloc[-1]
    predicted_price = predict_price(df)

    if last_price < df['SMA_50'].iloc[-1] and df['RSI'].iloc[-1] < 30:
        return "📈 Buy Signal (Oversold + Below SMA50)"
    elif last_price > df['SMA_50'].iloc[-1] and df['RSI'].iloc[-1] > 70:
        return "📉 Sell Signal (Overbought + Above SMA50)"
    else:
        return "⚖️ Hold Signal (Neutral Condition)"

# Streamlit UI
st.title("📈 AI-Powered Stock Prediction & Trading Signals")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

if st.button("Analyze & Predict"):
    df = get_live_data(ticker)
    predicted_price = predict_price(df)
    trading_signal = get_trading_signal(df)
    
    st.subheader(f"🔮 Predicted Price: ${predicted_price:.2f}")
    st.subheader(f"🚦 Trading Signal: {trading_signal}")
    
    # Plot Stock Data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
    st.plotly_chart(fig)
