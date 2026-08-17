import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import google.generativeai as genai
import time

st.set_page_config(page_title="SafeBets Master Multi-Horizon Predictor", page_icon="📈", layout="wide")

# --- 1. SECURE CONFIGURATION ---
password = st.text_input("Enter Password", type="password")
if password != "admin123":
    st.warning("Please enter the password to access the dashboard.")
    st.stop()

# Initialize Gemini AI
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini = genai.GenerativeModel('gemini-2.5-flash')
    ai_enabled = True
except Exception:
    ai_enabled = False
    st.sidebar.error("Gemini API key not found in Streamlit Secrets.")

# --- 2. ASSET MAP ---
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

# --- 3. CACHED SENTIMENT ENGINE ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_news_sentiment(ticker_symbol, asset_name):
    if not ai_enabled:
        return 0.0
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if not news:
            return 0.0
            
        headlines = "\n".join([item['title'] for item in news[:3]])
        
        prompt = f"""
        You are a financial analyst. Read these news headlines for {asset_name}:
        {headlines}
        Output ONLY a single sentiment float between -1.0 (bearish) and 1.0 (bullish).
        Do not write words. Example: 0.40
        """
        
        response = gemini.generate_content(prompt)
        return float(response.text.strip())
    except Exception:
        return 0.0

# --- 4. OPTIMIZED QUANT ENGINE ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_exact_price_predictions(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="2y")
        if df.empty or len(df) < 60:
            return None, None, None
            
        df = df[['Close', 'Volume']].copy()
        
        # Technical Indicator Calculations
        df['Return'] = df['Close'].pct_change()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
        
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        df['Volume_Change'] = df['Volume'].pct_change()
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df = df.dropna() 
        
        features = ['Return', 'SMA_10', 'SMA_50', 'RSI', 'BB_Position', 'MACD', 'MACD_Hist', 'Volume_Change']
        latest_price = df['Close'].iloc[-1]
        X_today = df.iloc[[-1]][features]
        historical_data = df.iloc[:-1].copy()
        
        horizons = {'1D': 1, '7D': 7, '14D': 14, '30D': 30}
        results = {}
        
        for name, days in horizons.items():
            target_return = (historical_data['Close'].shift(-days) - historical_data['Close']) / historical_data['Close']
            
            valid_idx = historical_data.index[:-days]
            X = historical_data.loc[valid_idx, features]
            y = target_return.loc[valid_idx]
            
            split_index = int(len(X) * 0.8)
            X_train = X.iloc[:split_index]
            y_train = y.iloc[:split_index]
            
            # Fast-Fitting Voting Ensemble
            model = VotingRegressor(estimators=[
                ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, learning_rate=0.05, max_depth=3, n_estimators=30, n_jobs=1)),
                ('rf', RandomForestRegressor(n_estimators=30, random_state=42, max_depth=3, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=30, random_state=42, learning_rate=0.05, max_depth=3))
            ])
            model.fit(X_train, y_train)
            
            pred_return = model.predict(X_today)[0]
            predicted_price = latest_price * (1 + pred_return)
            
            results[name] = {
                'predicted_price': predicted_price,
                'percent_change': pred_return * 100
            }
            
        return results, latest_price, df
    except Exception:
        return None, None, None

# --- 5. DASHBOARD INTERFACE ---
st.title("📈 SafeBets Master Prediction Table")
st.markdown("Generates **Math Ensemble Price Targets** adjusted by **Gemini AI News Sentiment**.")

if st.button("🚀 Run Fast All-Assets Analysis"):
    master_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_assets = len(asset_map)
    
    for idx, (asset_name, ticker) in enumerate(asset_map.items()):
        status_text.text(f"Processing ({idx + 1}/{total_assets}): {asset_name}")
        
        # 1. Quant Predictions
        results, price, _ = get_exact_price_predictions(ticker)
        
        # 2. Gemini News Sentiment
        sentiment_score = get_news_sentiment(ticker, asset_name)
        
        if results is not None:
            sentiment_text = f"🟢 +{sentiment_score:.2f}" if sentiment_score > 0.1 else f"🔴 {sentiment_score:.2f}" if sentiment_score < -0.1 else f"⚪ {sentiment_score:.2f}"
            
            sentiment_weight = 0.015 
            adjusted_targets = {}
            
            for h in ['1D', '7D', '14D', '30D']:
                raw_target = results[h]['predicted_price']
                adjusted_target = raw_target * (1 + (sentiment_score * sentiment_weight))
                adjusted_pct = ((adjusted_target - price) / price) * 100
                adjusted_targets[h] = f"${adjusted_target:,.2f} ({adjusted_pct:+.2f}%)"
            
            master_rows.append({
                "Asset": asset_name,
                "Current Price": f"${price:,.2f}",
                "News Sentiment": sentiment_text,
                "1D Target": adjusted_targets['1D'],
                "7D Target": adjusted_targets['7D'],
                "14D Target": adjusted_targets['14D'],
                "30D Target": adjusted_targets['30D'],
            })
        
        progress_bar.progress((idx + 1) / total_assets)
        time.sleep(1)  # Minimal 1-second pause to prevent API rate limits
        
    status_text.text("Analysis complete!")
    st.success("Master Market Sweep Complete!")
    
    if master_rows:
        summary_df = pd.DataFrame(master_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)