import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SafeBets Master Multi-Horizon Predictor", page_icon="📈", layout="wide")

# --- 1. SECURE CONFIGURATION (NO STARTUP API CALLS) ---
password = st.text_input("Enter Password", type="password")
if password != st.secrets.get("APP_PASSWORD", "admin123"):
    st.warning("Please enter the password to access the dashboard.")
    st.stop()

SENTIMENT_MODEL = "claude-haiku-4-5-20251001"
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
MAX_HEADLINES = 4

HISTORY_PERIOD = "5y"
N_FOLDS = 3
HALFLIFE_DAYS = 252          # recent data weighted more heavily
BAND_Z = 1.28                # ~80% prediction interval
MAX_SIGMA_CLIP = 3.0         # clip predicted return to +/- 3 horizon-sigmas

ANTHROPIC_KEY = str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
GEMINI_KEY = str(st.secrets.get("GEMINI_API_KEY", "")).strip()

if ANTHROPIC_KEY:
    st.sidebar.success(f"✅ Sentiment: Claude ({SENTIMENT_MODEL})")
elif GEMINI_KEY:
    st.sidebar.warning(f"⚠️ Sentiment: Gemini fallback ({GEMINI_FALLBACK_MODEL})")
else:
    st.sidebar.error("❌ No ANTHROPIC_API_KEY or GEMINI_API_KEY in Secrets")

st.sidebar.markdown("---")
apply_shrinkage = st.sidebar.checkbox(
    "Shrink targets by measured skill", value=True,
    help="Scales each prediction by its out-of-sample skill score. When a model "
         "has no measured edge, its target collapses to the current price."
)
show_bands = st.sidebar.checkbox("Show prediction ranges", value=True)
n_folds = st.sidebar.slider(
    "Validation folds", 0, 4, N_FOLDS, 1,
    help="Walk-forward folds used to measure skill. 0 skips validation entirely "
         "(much faster, but then there is no skill estimate and no shrinkage). "
         "A full 26-asset sweep takes roughly 2 minutes per fold."
)
sentiment_weight = st.sidebar.slider(
    "Sentiment weight", 0.0, 0.05, 0.015, 0.005,
    help="Max fractional nudge applied to a target at sentiment = +/-1.0."
)

if n_folds == 0 and apply_shrinkage:
    st.sidebar.info("Shrinkage needs at least 1 validation fold — it is inactive.")

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

# --- 3. BATCHED SENTIMENT ENGINE (ONE API CALL FOR ALL ASSETS) ---


def _fetch_headlines(ticker_symbol, limit=MAX_HEADLINES):
    """Pull up to `limit` headline strings, tolerating both yfinance news schemas."""
    try:
        news = yf.Ticker(ticker_symbol).news or []
    except Exception:
        return []

    out = []
    for item in news:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        title = item.get("title") or content.get("title") or item.get("headline")
        if title:
            out.append(str(title).strip())
        if len(out) >= limit:
            break
    return out


def _parse_json_scores(text):
    """Parse the model's JSON reply, tolerating code fences and stray prose."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _call_llm(prompt):
    """Return (text, status). Claude primary, Gemini fallback."""
    last_error = "No provider configured"

    if ANTHROPIC_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            msg = client.messages.create(
                model=SENTIMENT_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in msg.content
                if getattr(block, "type", "") == "text"
            )
            if text.strip():
                return text, "OK"
            last_error = "Claude returned empty response"
        except Exception as e:
            last_error = f"Claude: {str(e)[:60]}"

    if GEMINI_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel(GEMINI_FALLBACK_MODEL)
            resp = model.generate_content(prompt)
            if resp and resp.text and resp.text.strip():
                return resp.text, "OK"
            last_error = "Gemini returned empty response"
        except Exception as e:
            last_error = f"Gemini: {str(e)[:60]}"

    return None, last_error


@st.cache_data(ttl=1800, show_spinner=False)
def get_all_sentiments(asset_items):
    """
    asset_items: tuple of (asset_name, ticker_symbol) pairs.
    Returns {asset_name: (score, status)} using exactly ONE model call.
    """
    results = {name: (0.0, "No news found") for name, _ in asset_items}

    headline_map = {}
    for name, ticker in asset_items:
        headlines = _fetch_headlines(ticker)
        if headlines:
            headline_map[name] = headlines

    if not headline_map:
        return results

    if not (ANTHROPIC_KEY or GEMINI_KEY):
        return {name: (0.0, "No API key") for name, _ in asset_items}

    blocks = []
    for name, headlines in headline_map.items():
        joined = "\n".join(f"  - {h}" for h in headlines)
        blocks.append(f'"{name}":\n{joined}')

    prompt = (
        "You are a financial news sentiment scorer.\n"
        "For EACH asset below, score its headlines from -1.0 (very bearish) to "
        "1.0 (very bullish). Use 0.0 when the headlines are neutral or not "
        "actually relevant to that asset's price.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn ONLY a JSON object mapping each asset name, written exactly "
          "as it appears above, to its numeric score. No markdown, no code fences, "
          "no commentary.\n"
          'Example: {"Crypto - BTC": 0.35, "Tech - NVDA": -0.2}'
    )

    raw, status = _call_llm(prompt)
    if raw is None:
        return {name: (0.0, status) for name, _ in asset_items}

    parsed = _parse_json_scores(raw)
    if not parsed:
        return {name: (0.0, "Unparseable response") for name, _ in asset_items}

    for name in headline_map:
        if name in parsed:
            try:
                score = float(parsed[name])
                results[name] = (max(-1.0, min(1.0, score)), "OK")
            except (TypeError, ValueError):
                results[name] = (0.0, "Bad score value")
        else:
            results[name] = (0.0, "Missing from response")

    return results


# --- 4. QUANT ENGINE ---

FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_21', 'ret_63',
    'sma10_rel', 'sma50_rel', 'sma_cross',
    'rsi', 'bb_pos',
    'macd_rel', 'macd_hist_rel',
    'vol_21', 'vol_ratio', 'volume_rel', 'dist_52w_high',
]


def build_features(df):
    """
    Turn an OHLCV frame into scale-free (stationary) features.

    Every feature is a ratio, a bounded oscillator or a return. None of them
    carry the price level, so a model trained at $30k BTC stays valid at $90k.
    Raw SMA levels were the single biggest defect in the original version:
    tree models cannot extrapolate beyond their training range, so any asset
    trading above its historical band had its prediction silently clamped.
    """
    out = df[['Close', 'Volume']].copy()
    close = out['Close']

    out['ret_1'] = close.pct_change()
    out['ret_5'] = close.pct_change(5)
    out['ret_10'] = close.pct_change(10)
    out['ret_21'] = close.pct_change(21)
    out['ret_63'] = close.pct_change(63)

    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    out['sma10_rel'] = close / sma_10 - 1.0
    out['sma50_rel'] = close / sma_50 - 1.0
    out['sma_cross'] = sma_10 / sma_50 - 1.0

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out['rsi'] = (100 - (100 / (1 + rs))) / 100.0

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + std_20 * 2
    bb_lower = sma_20 - std_20 * 2
    out['bb_pos'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out['macd_rel'] = macd / close
    out['macd_hist_rel'] = (macd - macd_signal) / close

    out['vol_21'] = out['ret_1'].rolling(21).std()
    vol_63 = out['ret_1'].rolling(63).std()
    out['vol_ratio'] = out['vol_21'] / (vol_63 + 1e-9)

    vol_mean_20 = out['Volume'].rolling(20).mean()
    out['volume_rel'] = np.log((out['Volume'] + 1.0) / (vol_mean_20 + 1.0))

    out['dist_52w_high'] = close / close.rolling(252, min_periods=60).max() - 1.0

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _sample_weights(n, halflife=HALFLIFE_DAYS):
    """Exponential recency weights: the newest row weighs 1.0, older rows decay."""
    age = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (age / halflife)


def _fit_ensemble(X_train, y_train, weights):
    """
    Fit four models spanning different inductive biases and return a predictor.

    Ridge is included deliberately: financial return data has a very low
    signal-to-noise ratio, where a regularised linear model is usually more
    robust than boosted trees, and unlike trees it can extrapolate.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_scaled, y_train, sample_weight=weights)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', random_state=42, learning_rate=0.03,
        max_depth=3, n_estimators=120, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, n_jobs=1
    )
    xgb_model.fit(X_train, y_train, sample_weight=weights)

    rf_model = RandomForestRegressor(
        n_estimators=120, random_state=42, max_depth=4,
        min_samples_leaf=20, n_jobs=-1
    )
    rf_model.fit(X_train, y_train, sample_weight=weights)

    gb_model = GradientBoostingRegressor(
        n_estimators=120, random_state=42, learning_rate=0.03,
        max_depth=3, subsample=0.8, min_samples_leaf=20
    )
    gb_model.fit(X_train, y_train, sample_weight=weights)

    def predict(X_new):
        preds = np.column_stack([
            ridge.predict(scaler.transform(X_new)),
            xgb_model.predict(X_new),
            rf_model.predict(X_new),
            gb_model.predict(X_new),
        ])
        return preds.mean(axis=1)

    return predict


def walk_forward_skill(X, y, days, n_folds=N_FOLDS):
    """
    Measure genuine out-of-sample performance with a purge gap.

    The label at row t is built from prices at t+days, so any test row within
    `days` of the training cut is contaminated by data the model already saw.
    Purging that gap is what separates a real accuracy estimate from a
    flattering one.

    Returns (skill, directional_accuracy, n_tested) where skill is
    1 - MSE(model)/MSE(always predict zero). Positive means the model beats a
    random walk; zero or negative means it does not.
    """
    n = len(X)
    if n_folds < 1 or n < 250:
        return 0.0, 0.5, 0

    fold_size = n // (n_folds + 1)
    if fold_size <= days + 30:
        return 0.0, 0.5, 0

    all_preds, all_actual = [], []

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + days          # purge gap
        test_end = min(test_start + fold_size, n)

        if train_end < 120 or test_end - test_start < 20:
            continue

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X.iloc[test_start:test_end]
        y_te = y.iloc[test_start:test_end]

        try:
            predictor = _fit_ensemble(X_tr, y_tr, _sample_weights(len(X_tr)))
            all_preds.append(predictor(X_te))
            all_actual.append(y_te.to_numpy())
        except Exception:
            continue

    if not all_preds:
        return 0.0, 0.5, 0

    preds = np.concatenate(all_preds)
    actual = np.concatenate(all_actual)

    mse_model = float(np.mean((preds - actual) ** 2))
    mse_baseline = float(np.mean(actual ** 2))       # baseline: predict no change
    skill = 0.0 if mse_baseline <= 0 else 1.0 - (mse_model / mse_baseline)

    moved = np.abs(actual) > 1e-9
    dir_acc = float(np.mean(np.sign(preds[moved]) == np.sign(actual[moved]))) if moved.any() else 0.5

    return float(skill), dir_acc, int(len(preds))


@st.cache_data(ttl=1800, show_spinner=False)
def get_exact_price_predictions(ticker_symbol, shrink_enabled=True, folds=N_FOLDS):
    try:
        raw = yf.Ticker(ticker_symbol).history(period=HISTORY_PERIOD)
        if raw.empty or len(raw) < 300:
            return None, None, None

        df = build_features(raw).dropna()
        if len(df) < 300:
            return None, None, None

        latest_price = float(df['Close'].iloc[-1])
        X_today = df.iloc[[-1]][FEATURES]
        history = df.iloc[:-1].copy()

        daily_sigma = float(history['ret_1'].tail(126).std())

        horizons = {'1D': 1, '7D': 7, '14D': 14, '30D': 30}
        results = {}

        for name, days in horizons.items():
            target = history['Close'].shift(-days) / history['Close'] - 1.0

            usable = history.index[:-days]
            X = history.loc[usable, FEATURES]
            y = target.loc[usable]

            keep = y.notna()
            X, y = X[keep], y[keep]
            if len(X) < 250:
                continue

            skill, dir_acc, n_tested = walk_forward_skill(X, y, days, folds)

            # Final model trains on ALL available history, not a stale 80%.
            predictor = _fit_ensemble(X, y, _sample_weights(len(X)))
            pred_return = float(predictor(X_today)[0])

            # Keep the point estimate inside the asset's own realised range.
            horizon_sigma = daily_sigma * np.sqrt(days)
            cap = MAX_SIGMA_CLIP * horizon_sigma
            pred_return = float(np.clip(pred_return, -cap, cap))

            # With no validation folds there is no measurement, so there is no
            # basis to shrink. Zeroing everything there would be a false signal,
            # not a conservative one.
            if shrink_enabled and folds >= 1:
                shrink = float(np.clip(skill, 0.0, 1.0))
            else:
                shrink = 1.0
            effective_return = pred_return * shrink

            results[name] = {
                'predicted_price': latest_price * (1 + effective_return),
                'raw_return': pred_return,
                'effective_return': effective_return,
                'percent_change': effective_return * 100,
                'skill': skill,
                'dir_acc': dir_acc,
                'n_tested': n_tested,
                'sigma': horizon_sigma,
            }

        if not results:
            return None, None, None

        return results, latest_price, df
    except Exception:
        return None, None, None


# --- 5. DASHBOARD INTERFACE ---
st.title("📈 SafeBets Master Prediction Table")
st.markdown("Generates **Math Ensemble Price Targets** adjusted by **AI News Sentiment**.")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

if st.button("🚀 Run All-Assets Analysis"):
    master_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_assets = len(asset_map)

    status_text.text("Scoring news sentiment for all assets (single API call)...")
    sentiments = get_all_sentiments(tuple(asset_map.items()))

    for idx, (asset_name, ticker) in enumerate(asset_map.items()):
        status_text.text(f"Processing ({idx + 1}/{total_assets}): {asset_name}")

        results, price, _ = get_exact_price_predictions(ticker, apply_shrinkage, n_folds)
        sentiment_score, sentiment_status = sentiments.get(asset_name, (0.0, "n/a"))

        if results is not None:
            if sentiment_status == "OK":
                sentiment_text = (
                    f"🟢 +{sentiment_score:.2f}" if sentiment_score > 0.1
                    else f"🔴 {sentiment_score:.2f}" if sentiment_score < -0.1
                    else f"⚪ {sentiment_score:.2f}"
                )
            else:
                sentiment_text = f"⚠️ 0.00 ({sentiment_status})"

            row = {
                "Asset": asset_name,
                "Current Price": f"${price:,.2f}",
                "News Sentiment": sentiment_text,
            }

            skills = []
            for h in ['1D', '7D', '14D', '30D']:
                if h not in results:
                    row[f"{h} Target"] = "n/a"
                    continue

                r = results[h]
                skills.append(r['skill'])

                adjusted = r['predicted_price'] * (1 + sentiment_score * sentiment_weight)
                adjusted_pct = ((adjusted - price) / price) * 100

                cell = f"${adjusted:,.2f} ({adjusted_pct:+.2f}%)"
                if show_bands:
                    band = BAND_Z * r['sigma'] * price
                    cell += f"  ±${band:,.2f}"
                row[f"{h} Target"] = cell

            best_skill = max(skills) if skills else 0.0
            dir_acc_30 = results.get('30D', {}).get('dir_acc', 0.5)

            if best_skill > 0.01:
                row["Model Skill"] = f"🟢 {best_skill:+.3f}"
            elif best_skill > -0.05:
                row["Model Skill"] = f"⚪ {best_skill:+.3f}"
            else:
                row["Model Skill"] = f"🔴 {best_skill:+.3f}"

            row["30D Dir. Acc"] = f"{dir_acc_30:.0%}"
            master_rows.append(row)

        progress_bar.progress((idx + 1) / total_assets)

    status_text.text("Analysis complete!")
    st.success("Master Market Sweep Complete!")

    if master_rows:
        summary_df = pd.DataFrame(master_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            "**Reading the table.** *Model Skill* is an out-of-sample score: "
            "`1 - MSE(model) / MSE(assume no change)`, measured on data the model "
            "never trained on, with a purge gap so the label window cannot leak. "
            "Above 0 means it beat a random walk on that asset. At or below 0 means "
            "it did not, and with shrinkage enabled its target collapses toward the "
            "current price. *Dir. Acc* is how often the 30-day direction was right — "
            "50% is a coin flip. The ± figure is an 80% range from realised "
            "volatility, and it will usually dwarf the predicted move."
        )