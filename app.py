"""
SafeBets Multi-Horizon Predictor.

Built for a specific scoring rule: an exact price guess, scored on percentage
error. That rule dictates the whole design.

  * Percentage error on a price is, to first order, absolute error on its log.
    So the target is the log return and every metric lives in log space.
  * Absolute-error loss is minimised by the conditional MEDIAN, not the mean.
    Every learner here is a median/MAE learner, and members are combined by
    median rather than mean. Optimising squared error would target the mean,
    which sits above the median by roughly sigma^2/2 -- about 3% on a 30-day
    crypto horizon -- and would bias every submission upward.
  * "Just enter today's price" is the baseline to beat. Skill is measured
    against exactly that, and predictions are shrunk toward it in proportion
    to measured skill, because under MAE a forecast you cannot justify costs
    you points.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import re
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SafeBets Multi-Horizon Predictor", page_icon="📈", layout="wide")

# --- 1. CONFIGURATION ---
password = st.text_input("Enter Password", type="password")
if password != st.secrets.get("APP_PASSWORD", "admin123"):
    st.warning("Please enter the password to access the dashboard.")
    st.stop()

SENTIMENT_MODEL = "claude-haiku-4-5-20251001"
GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
MAX_HEADLINES = 4

HISTORY_PERIOD = "5y"
DEFAULT_FOLDS = 3
HALFLIFE_DAYS = 252
BAND_Z = 1.28
MAX_SIGMA_CLIP = 2.0   # tighter than before: under MAE, bold calls are punished

HORIZONS = {'1D': 1, '7D': 7, '14D': 14, '30D': 30}

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
    "Shrink toward today's price by measured skill", value=True,
    help="Under percentage-error scoring this is almost always correct. A model "
         "with no measured edge should submit today's price."
)
n_folds = st.sidebar.slider(
    "Validation folds", 1, 4, DEFAULT_FOLDS, 1,
    help="Walk-forward folds used to measure skill against the 'enter today's "
         "price' baseline. Roughly 1.5 minutes per fold for a full sweep."
)
sentiment_weight = st.sidebar.slider(
    "Sentiment weight", 0.0, 0.03, 0.0, 0.0025,
    help="Defaults to 0. A sentiment nudge only helps if it beats the baseline, "
         "and that has not been validated. Raise it only after backtesting."
)
show_bands = st.sidebar.checkbox("Show 80% ranges", value=True)

# --- 2. ASSET MAP ---
asset_map = {
    "Crypto - BTC": "BTC-USD", "Crypto - ETH": "ETH-USD", "Crypto - SOL": "SOL-USD",
    "Crypto - DOGE": "DOGE-USD", "Crypto - AVAX": "AVAX-USD", "Crypto - LINK": "LINK-USD",
    "Tech - NVDA": "NVDA", "Tech - TSLA": "TSLA", "Tech - AAPL": "AAPL", "Tech - MSFT": "MSFT",
    "Tech - AMZN": "AMZN", "Tech - META": "META", "Tech - GOOGL": "GOOGL", "Tech - NFLX": "NFLX",
    "Chips - AMD": "AMD", "Chips - MU": "MU", "Chips - AVGO": "AVGO",
    "Chips - INTC": "INTC", "Chips - ARM": "ARM",
    "Comm - GOLD": "GC=F", "Comm - SILVER": "SI=F", "Comm - WTI": "CL=F", "Comm - COPPER": "HG=F"
}

# --- 3. BATCHED SENTIMENT ENGINE (ONE API CALL FOR ALL ASSETS) ---


def _fetch_headlines(ticker_symbol, limit=MAX_HEADLINES):
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
    Scale-free features only. Nothing carries the price level, so a model
    trained at $30k BTC stays valid at $90k. Raw SMA levels were the original
    defect: trees cannot extrapolate past their training range, so any asset
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

    out['log_ret'] = np.log(close).diff()
    out['vol_21'] = out['log_ret'].rolling(21).std()
    vol_63 = out['log_ret'].rolling(63).std()
    out['vol_ratio'] = out['vol_21'] / (vol_63 + 1e-9)

    vol_mean_20 = out['Volume'].rolling(20).mean()
    out['volume_rel'] = np.log((out['Volume'] + 1.0) / (vol_mean_20 + 1.0))

    out['dist_52w_high'] = close / close.rolling(252, min_periods=60).max() - 1.0

    return out.replace([np.inf, -np.inf], np.nan)


def _sample_weights(n, halflife=HALFLIFE_DAYS):
    age = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (age / halflife)


def fit_median_ensemble(X_train, y_train, weights):
    """
    Three MAE/median learners with different inductive biases, combined by
    median. Every one of them targets the conditional median of the log
    return, which is the quantity that minimises percentage error.
    """
    members = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    try:
        qr = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
        qr.fit(X_scaled, y_train, sample_weight=weights)
        members.append(lambda Xn, _qr=qr, _s=scaler: _qr.predict(_s.transform(Xn)))
    except Exception:
        pass

    try:
        xgb_model = xgb.XGBRegressor(
            objective='reg:absoluteerror', random_state=42, learning_rate=0.03,
            max_depth=3, n_estimators=150, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, n_jobs=1
        )
        xgb_model.fit(X_train, y_train, sample_weight=weights)
        members.append(lambda Xn, _m=xgb_model: _m.predict(Xn))
    except Exception:
        pass

    try:
        hgb = HistGradientBoostingRegressor(
            loss='absolute_error', max_iter=150, learning_rate=0.03,
            max_depth=3, min_samples_leaf=25, l2_regularization=1.0,
            random_state=42
        )
        hgb.fit(X_train, y_train, sample_weight=weights)
        members.append(lambda Xn, _m=hgb: _m.predict(Xn))
    except Exception:
        pass

    if not members:
        return lambda Xn: np.zeros(len(Xn))

    def predict(X_new):
        stacked = np.column_stack([m(X_new) for m in members])
        return np.median(stacked, axis=1)   # median combine, matching the loss

    return predict


def walk_forward_mae(X, y, days, n_folds):
    """
    Out-of-sample MAE against the only baseline that matters here: submitting
    today's price, i.e. predicting a log return of zero.

    The label at row t is built from prices at t+days, so a test row within
    `days` of the training cut is contaminated. That gap is purged.

    Returns (skill, mae_model, mae_baseline, n_tested) where
    skill = 1 - MAE(model)/MAE(baseline). Positive means the model beat
    "enter today's price"; zero or below means it did not.
    """
    n = len(X)
    if n_folds < 1 or n < 250:
        return 0.0, np.nan, np.nan, 0

    fold_size = n // (n_folds + 1)
    if fold_size <= days + 30:
        return 0.0, np.nan, np.nan, 0

    preds, actuals = [], []

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + days
        test_end = min(test_start + fold_size, n)

        if train_end < 120 or test_end - test_start < 20:
            continue

        try:
            predictor = fit_median_ensemble(
                X.iloc[:train_end], y.iloc[:train_end], _sample_weights(train_end)
            )
            preds.append(predictor(X.iloc[test_start:test_end]))
            actuals.append(y.iloc[test_start:test_end].to_numpy())
        except Exception:
            continue

    if not preds:
        return 0.0, np.nan, np.nan, 0

    p = np.concatenate(preds)
    a = np.concatenate(actuals)

    mae_model = float(np.mean(np.abs(p - a)))
    mae_base = float(np.mean(np.abs(a)))       # baseline: predict zero log return
    skill = 0.0 if mae_base <= 0 else 1.0 - (mae_model / mae_base)

    return float(skill), mae_model, mae_base, int(len(p))


@st.cache_data(ttl=1800, show_spinner=False)
def get_predictions(ticker_symbol, shrink_enabled=True, folds=DEFAULT_FOLDS):
    try:
        raw = yf.Ticker(ticker_symbol).history(period=HISTORY_PERIOD)
        if raw.empty or len(raw) < 300:
            return None, None

        df = build_features(raw).dropna()
        if len(df) < 300:
            return None, None

        latest_price = float(df['Close'].iloc[-1])
        X_today = df.iloc[[-1]][FEATURES]
        history = df.iloc[:-1].copy()
        log_close = np.log(history['Close'])

        daily_sigma = float(history['log_ret'].tail(126).std())

        results = {}

        for name, days in HORIZONS.items():
            # Target is the log return over the horizon: percentage error on a
            # price is absolute error on its log, to first order.
            target = log_close.shift(-days) - log_close

            usable = history.index[:-days]
            X = history.loc[usable, FEATURES]
            y = target.loc[usable]

            keep = y.notna()
            X, y = X[keep], y[keep]
            if len(X) < 250:
                continue

            skill, mae_model, mae_base, n_tested = walk_forward_mae(X, y, days, folds)

            predictor = fit_median_ensemble(X, y, _sample_weights(len(X)))
            pred_log = float(predictor(X_today)[0])

            horizon_sigma = daily_sigma * np.sqrt(days)
            cap = MAX_SIGMA_CLIP * horizon_sigma
            pred_log = float(np.clip(pred_log, -cap, cap))

            shrink = float(np.clip(skill, 0.0, 1.0)) if shrink_enabled else 1.0
            effective_log = pred_log * shrink

            results[name] = {
                'price': latest_price * float(np.exp(effective_log)),
                'pct': (float(np.exp(effective_log)) - 1.0) * 100.0,
                'raw_log': pred_log,
                'skill': skill,
                'mae_model_pct': mae_model * 100 if np.isfinite(mae_model) else np.nan,
                'mae_base_pct': mae_base * 100 if np.isfinite(mae_base) else np.nan,
                'n_tested': n_tested,
                'lo': latest_price * float(np.exp(effective_log - BAND_Z * horizon_sigma)),
                'hi': latest_price * float(np.exp(effective_log + BAND_Z * horizon_sigma)),
            }

        if not results:
            return None, None

        return results, latest_price
    except Exception:
        return None, None


# --- 5. DASHBOARD ---
st.title("📈 SafeBets Multi-Horizon Predictor")
st.caption(
    "Optimised for percentage-error scoring: every target is a conditional "
    "median log return, and skill is measured against submitting today's price."
)

if st.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

if st.button("🚀 Run All-Assets Analysis"):
    submission_rows, diagnostic_rows = [], []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(asset_map)

    if sentiment_weight > 0:
        status_text.text("Scoring news sentiment (single API call)...")
        sentiments = get_all_sentiments(tuple(asset_map.items()))
    else:
        sentiments = {name: (0.0, "disabled") for name in asset_map}

    for idx, (asset_name, ticker) in enumerate(asset_map.items()):
        status_text.text(f"Processing ({idx + 1}/{total}): {asset_name}")

        results, price = get_predictions(ticker, apply_shrinkage, n_folds)
        sent_score, sent_status = sentiments.get(asset_name, (0.0, "n/a"))

        if results:
            sub = {"Asset": asset_name, "Current": f"${price:,.4f}" if price < 1 else f"${price:,.2f}"}
            diag = {"Asset": asset_name}

            for h in HORIZONS:
                if h not in results:
                    sub[h] = "n/a"
                    diag[f"{h} skill"] = "n/a"
                    continue

                r = results[h]
                final = r['price'] * (1 + sent_score * sentiment_weight)
                pct = (final / price - 1.0) * 100.0

                cell = f"${final:,.4f}" if final < 1 else f"${final:,.2f}"
                cell += f"  ({pct:+.2f}%)"
                if show_bands:
                    lo, hi = r['lo'], r['hi']
                    fmt = "{:,.4f}" if hi < 1 else "{:,.2f}"
                    cell += f"\n[{fmt.format(lo)} – {fmt.format(hi)}]"
                sub[h] = cell

                mark = "🟢" if r['skill'] > 0.005 else ("⚪" if r['skill'] > -0.02 else "🔴")
                diag[f"{h} skill"] = f"{mark} {r['skill']:+.3f}"
                diag[f"{h} err"] = (
                    f"{r['mae_model_pct']:.2f}% vs {r['mae_base_pct']:.2f}%"
                    if np.isfinite(r['mae_model_pct']) else "n/a"
                )

            submission_rows.append(sub)
            diagnostic_rows.append(diag)

        progress_bar.progress((idx + 1) / total)

    status_text.text("Analysis complete!")

    if submission_rows:
        st.subheader("Values to submit")
        st.dataframe(pd.DataFrame(submission_rows), use_container_width=True, hide_index=True)

        st.subheader("Did the model beat 'just enter today's price'?")
        st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)

        st.markdown(
            "**Skill** is `1 - MAE(model) / MAE(enter today's price)`, measured "
            "out-of-sample with a purge gap. **Err** shows the model's mean "
            "absolute percentage error next to the baseline's. Green means the "
            "model genuinely beat the baseline on that asset and horizon; grey "
            "or red means it did not, and with shrinkage on, that target has "
            "collapsed to today's price — which under percentage-error scoring "
            "is the correct submission, not a failure. Expect most cells to be "
            "grey. The 1D column is the hardest to beat and the 30D column the "
            "easiest, because the baseline's own error grows with the horizon."
        )