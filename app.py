import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import google.generativeai as genai

st.set_page_config(page_title="SafeBets Multi-Horizon AI", page_icon="📈", layout="wide")

# --- 1. SECURE CONFIGURATION ---
password = st.text_input("Enter Password", type="password")
if password != "admin123":
    st.warning("Please enter the password to access the dashboard.")
    st.stop()

# Initialize Gemini AI
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini = genai.GenerativeModel('gemini-2.5-flash') # Using the fast, free tier model
    ai_enabled = True
except Exception as e:
    ai_enabled = False
    st.sidebar.error("Gemini API not configured properly in Secrets.")

# --- 2. ASSET MAP ---
asset_map = {
    "Crypto - BTC": "BTC-USD", "Crypto - ETH": "ETH-USD", "Crypto - SOL": "SOL-USD", 
    "Crypto - DOGE": "DOGE-USD", "Crypto - AVAX": "AVAX-USD", "Crypto - LINK": "LINK-USD", "Crypto - HYPE": "HYPE-USD",
    "Tech - NVDA": "NVDA", "Tech - TSLA": "TSLA", "Tech - AAPL": "AAPL", "Tech - MSFT": "MSFT", 
    "Tech - AMZN": "AMZN", "Tech - META": "META", "Tech - GOOGL": "GOOGL", "Tech - NFLX": "NFLX", "Tech - SPCX": "SPCX",
    "Chips - AMD": "AMD", "Chips - MU": "MU", "Chips - SNDK": "SNDK", "Chips - AVGO": "AVGO", 
    "Chips - INTC": "INTC", "Chips - ARM": "ARM",
    "Comm - GOLD": "GC=F", "Comm - SILVER": "SI=F", "Comm - WTI": "CL=F", "Comm - COPPER": "HG=F"
}

# --- 3. NLP SENTIMENT ENGINE ---
@st.cache_data(ttl=3600)
def get_news_sentiment(ticker_symbol, asset_name):
    if not ai_enabled:
        return 0.0, "AI Sentiment disabled (check API key)."
    
    try:
        # Fetch the top 5 recent news headlines for the asset
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if not news:
            return 0.0, "No recent news found for this asset."
            
        headlines = "\n".join([item['title'] for item in news[:5]])
        
        # Prompt Gemini to score the headlines
        prompt = f"""
        You are an expert quantitative financial analyst. 
        Read the following recent news headlines for {asset_name}:
        {headlines}
        
        Provide a sentiment score between -1.0 (highly bearish/negative) and 1.0 (highly bullish/positive).
        Then provide a 1-sentence explanation of why.
        Format your response EXACTLY like this: [SCORE] | [EXPLANATION]
        Example: 0.8 | Strong earnings reports and new AI chip demand are driving positive momentum.
        """
        
        response = gemini.generate_content(prompt)
        parts = response.text.split('|')
        
        score = float(parts[0].strip())
        explanation = parts[1].strip()
        return score, explanation
        
    except Exception as e:
        return 0.0, f"Could not generate sentiment. Error: {e}"

# --- 4. QUANT PREDICTION ENGINE (Ensemble) ---
@st.cache_data(ttl=3600)
def get_exact_price_predictions(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="5y")
        if df.empty or len(df) < 100:
            return None, None, None
            
        df = df[['Close', 'Volume']].copy()
        
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
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            model = VotingRegressor(estimators=[
                ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, learning_rate=0.03, max_depth=4, n_estimators=100)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=4)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.03, max_depth=4))
            ])
            model.fit(X_train, y_train)
            
            test_preds = model.predict(X_test)
            dir_accuracy = accuracy_score(y_test > 0, test_preds > 0)
            avg_dollar_error = mean_absolute_error(y_test * latest_price, test_preds * latest_price)
            
            pred_return = model.predict(X_today)[0]
            predicted_price = latest_price * (1 + pred_return)
            dollar_change = predicted_price - latest_price
            
            results[name] = {
                'predicted_price': predicted_price,
                'dollar_change': dollar_change,
                'percent_change': pred_return * 100,
                'dir_accuracy': dir_accuracy,
                'avg_dollar_error': avg_dollar_error
            }
            
        return results, latest_price, df
    except Exception:
        return None, None, None

# --- 5. DASHBOARD INTERFACE ---
st.title("📈 SafeBets Master Multi-Horizon Predictor")

tab1, tab2 = st.tabs(["📊 Master Table (Quant Only)", "🔍 Detailed View (Quant + NLP Sentiment)"])

with tab1:
    st.subheader("Daily All-Assets Prediction Table")
    if st.button("🚀 Generate All Market Predictions"):
        master_rows = []
        progress_bar = st.progress(0)
        total_assets = len(asset_map)
        
        for idx, (asset_name, ticker) in enumerate(asset_map.items()):
            results, price, _ = get_exact_price_predictions(ticker)
            if results is not None:
                master_rows.append({
                    "Asset": asset_name,
                    "Current Price": f"${price:,.2f}",
                    "1D Target": f"${results['1D']['predicted_price']:,.2f}",
                    "7D Target": f"${results['7D']['predicted_price']:,.2f}",
                    "14D Target": f"${results['14D']['predicted_price']:,.2f}",
                    "30D Target": f"${results['30D']['predicted_price']:,.2f}",
                })
            progress_bar.progress((idx + 1) / total_assets)
            
        st.success("All quantitative predictions generated successfully!")
        if master_rows:
            st.dataframe(pd.DataFrame(master_rows), use_container_width=True, hide_index=True)

with tab2:
    selected_asset = st.selectbox("Select Asset for Detailed View", list(asset_map.keys()))
    ticker = asset_map[selected_asset]

    if st.button("Generate Detailed Prediction & Read News"):
        with st.spinner(f"Running Ensemble Math & Analyzing News for {selected_asset}..."):
            
            # Fetch Math & Sentiment Concurrently
            results, price, df = get_exact_price_predictions(ticker)
            sentiment_score, sentiment_summary = get_news_sentiment(ticker, selected_asset)
            
            if results is None:
                st.error("Not enough historical data available.")
                st.stop()
                
            st.divider()
            
            # --- DUAL SIGNAL DISPLAY ---
            score_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "gray"
            st.markdown(f"### 🧠 AI News Sentiment: :{score_color}[{sentiment_score}]")
            st.info(f"**Gemini Analysis:** {sentiment_summary}")
            
            st.metric("Current Price", f"${price:,.2f}")
            
            cols = st.columns(4)
            horizons = ['1D', '7D', '14D', '30D']
            
            for i, col in enumerate(cols):
                horizon = horizons[i]
                data = results[horizon]
                with col:
                    st.subheader(f"{horizon} Forecast")
                    st.metric(
                        label="Target Price", 
                        value=f"${data['predicted_price']:,.2f}", 
                        delta=f"{data['dollar_change']:+.2f} ({data['percent_change']:+.2f}%)"
                    )
                    st.caption(f"Quant Acc: {data['dir_accuracy'] * 100:.1f}%")
            
            st.divider()
            st.line_chart(df['Close'].tail(90))