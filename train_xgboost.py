import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# 1️⃣ Fetch Stock Data
def get_stock_data(ticker, start='2015-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

# 2️⃣ Preprocess Data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)
    
    X, Y = [], []
    time_step = 60
    for i in range(len(df_scaled) - time_step - 1):
        X.append(df_scaled[i:i + time_step, 0])
        Y.append(df_scaled[i + time_step, 0])

    X, Y = np.array(X), np.array(Y)
    return X, Y, scaler

# ✅ Train XGBoost Model
def train_xgboost(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
    model.fit(X_train, Y_train)
    
    return model

# Run Training Pipeline
ticker = "AAPL"
df = get_stock_data(ticker)
X, Y, scaler = preprocess_data(df)

# Train and Save the Model
xgb_model = train_xgboost(X, Y)
joblib.dump(xgb_model, "xgb_model.pkl")  # 🔥 Save model
joblib.dump(scaler, "scaler.pkl")  # 🔥 Save scaler

print("✅ XGBoost Model Trained & Saved as `xgb_model.pkl`!")
