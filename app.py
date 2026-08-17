import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

st.set_page_config(page_title="SafeBets Multi-Horizon AI", page_icon="📈", layout="wide")

# --- 1. PASSCODE SECURITY ---
password = st.text_input("Enter Password", type="password")
if password != "admin123":
    st.warning("Please enter the password to access the dashboard.")
    st.stop()

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

# --- 3. ENSEMBLE PREDICTION ENGINE ---
@st.cache_data(ttl=3600)  # Caches results for 1 hour to keep dashboard fast
def get_exact_price_predictions(ticker_symbol):
    try:
        df = yf.Ticker(ticker_symbol).history(period="5y")
        if df.empty or len(df) < 100:
            return None, None, None
            
        df = df[['Close', 'Volume']].copy()
        
        # Technical Indicator Feature Engineering
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
        
        latest_price = df['Close'].iloc[-1]
        X_today = df.iloc[[-1]][features]
        historical_data = df.iloc[:-1].copy()
        
        horizons = {'1D': 1, '7D': 7, '14D': 14, '30D': 30}
        results = {}
        
        for name, days in horizons.items():
            # Percentage return target over 'days'
            target_return = (historical_data['Close'].shift(-days) - historical_data['Close']) / historical_data['Close']
            
            valid_idx = historical_data.index[:-days]
            X = historical_data.loc[valid_idx, features]
            y = target_return.loc[valid_idx]
            
            split_index = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # --- 3-MODEL ENSEMBLE ---
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror', random_state=42, learning_rate=0.03, max_depth=4, n_estimators=100
            )
            rf_model = RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=4
            )
            gb_model = GradientBoostingRegressor(
                n_estimators=100, random_state=42, learning_rate=0.03, max_depth=4
            )
            
            # Voting Ensemble averages predictions from all 3 models
            model = VotingRegressor(estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model)
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

# --- 4. DASHBOARD INTERFACE ---
st.title("📈 SafeBets Master Multi-Horizon Predictor")

tab1, tab2 = st.tabs(["📊 Master Table (All Assets)", "🔍 Single Asset Detailed View"])

# --- TAB 1: ALL ASSETS AT ONCE ---
with tab1:
    st.subheader("Daily All-Assets Prediction Table")
    st.markdown("Click below to train ensemble models and generate targets for all assets simultaneously.")
    
    if st.button("🚀 Generate All Market Predictions"):
        master_rows = []
        progress_bar = st.progress(0)
        total_assets = len(asset_map)
        
        for idx, (asset_name, ticker) in enumerate(asset_map.items()):
            st.toast(f"Processing {asset_name}...", icon="⏳")
            results, price, _ = get_exact_price_predictions(ticker)
            
            if results is not None:
                master_rows.append({
                    "Asset": asset_name,
                    "Current Price": f"${price:,.2f}",
                    "1D Target": f"${results['1D']['predicted_price']:,.2f} ({results['1D']['percent_change']:+.2f}%)",
                    "7D Target": f"${results['7D']['predicted_price']:,.2f} ({results['7D']['percent_change']:+.2f}%)",
                    "14D Target": f"${results['14D']['predicted_price']:,.2f} ({results['14D']['percent_change']:+.2f}%)",
                    "30D Target": f"${results['30D']['predicted_price']:,.2f} ({results['30D']['percent_change']:+.2f}%)",
                })
            
            progress_bar.progress((idx + 1) / total_assets)
            
        st.success("All market predictions generated successfully!")
        
        if master_rows:
            summary_df = pd.DataFrame(master_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# --- TAB 2: SINGLE ASSET DETAILED VIEW ---
with tab2:
    selected_asset = st.selectbox("Select Asset for Detailed View", list(asset_map.keys()))
    ticker = asset_map[selected_asset]

    if st.button("Generate Detailed Prediction"):
        with st.spinner(f"Running ensemble analysis for {selected_asset}..."):
            results, price, df = get_exact_price_predictions(ticker)
            
            if results is None:
                st.error("Not enough historical data available for this symbol.")
                st.stop()
                
            st.divider()
            st.metric("Current Price", f"${price:,.2f}")
            
            cols = st.columns(4)
            horizons = ['1D', '7D', '14D', '30D']
            
            for i, col in enumerate(cols):
                horizon = horizons[i]
                data = results[horizon]
                
                target_p = data['predicted_price']
                d_change = data['dollar_change']
                p_change = data['percent_change']
                
                with col:
                    st.subheader(f"{horizon} Forecast")
                    st.metric(
                        label="Target Price", 
                        value=f"${target_p:,.2f}", 
                        delta=f"{d_change:+.2f} ({p_change:+.2f}%)"
                    )
                    st.caption(f"Directional Acc: {data['dir_accuracy'] * 100:.1f}%")
                    st.caption(f"Avg Error: ±${data['avg_dollar_error']:,.2f}")
            
            st.divider()
            st.subheader(f"90-Day Price History ({selected_asset})")
            st.line_chart(df['Close'].tail(90))