import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

# --- 1. SECURITY (Password Protection) ---
st.set_page_config(page_title="SafeBets AI Predictor", page_icon="📈")

password = st.text_input("Enter Password", type="password")
if password != "admin123":
    st.warning("Please enter the password to access the dashboard.")
    st.stop() # Stops the rest of the app from loading until the password is correct

# --- 2. AI MODEL FUNCTION ---
@st.cache_data # This tells Streamlit not to re-download the data every time you click a button
def get_prediction(ticker_symbol):
    df = yf.Ticker(ticker_symbol).history(period="5y")
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
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    historical_data = df.iloc[:-1]
    todays_data = df.iloc[[-1]]
    
    features = ['Return', 'SMA_10', 'SMA_50', 'RSI', 'BB_Position', 'MACD', 'MACD_Hist', 'Volume_Change']
    
    X = historical_data[features]
    y = historical_data['Target']
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    X_today = todays_data[features]
    prediction = model.predict(X_today)[0]
    probability = model.predict_proba(X_today)[0]
    
    latest_price = todays_data['Close'].iloc[-1]
    
    return accuracy, prediction, probability, latest_price, df

# --- 3. DASHBOARD UI ---
st.title("📈 AI Commodity Predictor")
st.markdown("Use these daily figures to enter into **SafeBets**.")

# Let the user choose the commodity
commodity_map = {"Gold": "GC=F", "Crude Oil": "CL=F", "Natural Gas": "NG=F"}
selected_commodity = st.selectbox("Select Commodity to Predict", list(commodity_map.keys()))
ticker = commodity_map[selected_commodity]

if st.button("Generate Today's Prediction"):
    with st.spinner(f"Analyzing data for {selected_commodity}..."):
        acc, pred, prob, price, df = get_prediction(ticker)
        
        st.divider()
        
        # Display large metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${price:.2f}")
        
        direction = "⬆️ UP" if pred == 1 else "⬇️ DOWN"
        conf = prob[1] if pred == 1 else prob[0]
        col2.metric("Tomorrow's Prediction", direction)
        col3.metric("AI Confidence", f"{conf * 100:.1f}%")
        
        st.caption(f"Model Backtest Historical Accuracy: {acc * 100:.1f}%")
        
        # Show a chart of the last 30 days
        st.subheader(f"30-Day Price Trend ({selected_commodity})")
        st.line_chart(df['Close'].tail(30))