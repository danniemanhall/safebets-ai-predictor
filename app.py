import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="SafeBets Multi-Horizon AI", page_icon="📈", layout="wide")

password = st.text_input("Enter Password", type="password")
if password != "admin123":
    st.warning("Please enter the password to access the dashboard.")
    st.stop()

# --- EXPANDED ASSET MAP ---
asset_map = {
    # Crypto
    "Crypto - BTC": "BTC-USD", "Crypto - ETH": "ETH-USD", "Crypto - SOL": "SOL-USD", 
    "Crypto - DOGE": "DOGE-USD", "Crypto - AVAX": "AVAX-USD", "Crypto - LINK": "LINK-USD", "Crypto - HYPE": "HYPE-USD",
    
    # Big Tech
    "Tech - NVDA": "NVDA", "Tech - TSLA": "TSLA", "Tech - AAPL": "AAPL", "Tech - MSFT": "MSFT", 
    "Tech - AMZN": "AMZN", "Tech - META": "META", "Tech - GOOGL": "GOOGL", "Tech - NFLX": "NFLX", "Tech - SPCX": "SPCX",
    
    # AI Chips
    "Chips - AMD": "AMD", "Chips - MU": "MU", "Chips - SNDK": "SNDK", "Chips - AVGO": "AVGO", 
    "Chips - INTC": "INTC", "Chips - ARM": "ARM",
    
    # Commodities
    "Comm - GOLD": "GC=F", "Comm - SILVER": "SI=F", "Comm - WTI": "CL=F", "Comm - COPPER": "HG=F"
}

@st.cache_data
def get_multi_horizon_predictions(ticker_symbol):
    df = yf.Ticker(ticker_symbol).history(period="5y")
    if df.empty:
        return None, None, None
        
    df = df[['Close', 'Volume']].copy()
    
    # Feature Engineering
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['Volume_Change'] = df['Volume'].pct_change()
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.dropna() 
    
    if len(df) < 100:
        return None, None, None 
        
    features = ['Return', 'SMA_10', 'SMA_50', 'RSI', 'BB_Position', 'MACD', 'MACD_Hist', 'Volume_Change']
    
    # Isolate today's data to predict the future
    X_today = df.iloc[[-1]][features]
    historical_data = df.iloc[:-1].copy()
    
    # The different time horizons we want to predict
    horizons = {'1D': 1, '7D': 7, '14D': 14, '30D': 30}
    results = {}
    
    for name, days in horizons.items():
        # Create target shifted by 'days'
        target = (historical_data['Close'].shift(-days) > historical_data['Close']).astype(int)
        
        # We must drop the last 'days' rows during training because the future hasn't happened yet
        valid_idx = historical_data.index[:-days]
        X = historical_data.loc[valid_idx, features]
        y = target.loc[valid_idx]
        
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # Train a brand new model just for this specific time horizon
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, learning_rate=0.05, max_depth=4)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        pred = model.predict(X_today)[0]
        prob = model.predict_proba(X_today)[0]
        
        results[name] = {
            'accuracy': acc,
            'prediction': pred,
            'confidence': prob[1] if pred == 1 else prob[0]
        }
        
    latest_price = df['Close'].iloc[-1]
    return results, latest_price, df

# --- DASHBOARD UI ---
st.title("📈 SafeBets Multi-Horizon Predictor")
st.markdown("Select an asset below to generate independent AI predictions for 1, 7, 14, and 30 days out.")

selected_asset = st.selectbox("Select Asset", list(asset_map.keys()))
ticker = asset_map[selected_asset]

if st.button("Generate Predictions"):
    with st.spinner(f"Training 4 AI models for {selected_asset}..."):
        results, price, df = get_multi_horizon_predictions(ticker)
        
        if results is None:
            st.error("Not enough historical data to generate predictions for this asset.")
            st.stop()
            
        st.divider()
        st.metric("Current Price", f"${price:,.2f}")
        
        # Display the 4 time horizons side-by-side
        cols = st.columns(4)
        horizons = ['1D', '7D', '14D', '30D']
        
        for i, col in enumerate(cols):
            horizon = horizons[i]
            data = results[horizon]
            
            direction = "⬆️ UP" if data['prediction'] == 1 else "⬇️ DOWN"
            
            with col:
                st.subheader(f"{horizon} Forecast")
                st.metric("Trend", direction)
                st.metric("Confidence", f"{data['confidence'] * 100:.1f}%")
                st.caption(f"Backtest Acc: {data['accuracy'] * 100:.1f}%")
        
        st.divider()
        st.subheader(f"90-Day Price Trend ({selected_asset})")
        st.line_chart(df['Close'].tail(90))