import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import joblib
import pandas_ta as ta  
from datetime import datetime, timedelta

# Load trained model and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to Fetch Live Stock Data
def get_live_data(ticker, days=90):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to Make Predictions
def predict_price(df):
    scaled_data = scaler.transform(df[['Close']])
    last_60_days = scaled_data[-60:].reshape(1, -1)
    predicted_price = xgb_model.predict(last_60_days)
    return scaler.inverse_transform([[predicted_price[0]]])[0][0]

# Function to Generate Trading Signals
def get_trading_signal(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  
    df['RSI'] = df.ta.rsi(length=14)  

    last_price = df['Close'].iloc[-1]
    predicted_price = predict_price(df)

    if last_price < df['SMA_50'].iloc[-1] and df['RSI'].iloc[-1] < 30:
        return "📈 Buy Signal (Oversold)"
    elif last_price > df['SMA_50'].iloc[-1] and df['RSI'].iloc[-1] > 70:
        return "📉 Sell Signal (Overbought)"
    else:
        return "⚖️ Hold Signal"

# Streamlit UI
st.title("📈 AI-Powered Stock Prediction (XGBoost)")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

if st.button("Analyze & Predict"):
    df = get_live_data(ticker)
    predicted_price = predict_price(df)
    trading_signal = get_trading_signal(df)
    
    st.subheader(f"🔮 Predicted Price: ${predicted_price:.2f}")
    st.subheader(f"🚦 Trading Signal: {trading_signal}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
    st.plotly_chart(fig)
