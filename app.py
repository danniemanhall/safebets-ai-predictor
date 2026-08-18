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
use_live = st.sidebar.checkbox(
    "Anchor on live price", value=True,
    help="Anchor every submission on the freshest price available rather than "
         "the daily close. For 24/7 assets the close can be hours old, and that "
         "gap is pure avoidable error."
)

# --- 2. ASSET MAP ---
asset_map = {
    # Crypto (7)
    "Crypto - BTC": "BTC-USD", "Crypto - ETH": "ETH-USD", "Crypto - SOL": "SOL-USD",
    "Crypto - DOGE": "DOGE-USD", "Crypto - AVAX": "AVAX-USD", "Crypto - LINK": "LINK-USD",
    "Crypto - HYPE": "HYPE32196-USD",

    # Big Tech (9)
    "Tech - NVDA": "NVDA", "Tech - TSLA": "TSLA", "Tech - AAPL": "AAPL", "Tech - MSFT": "MSFT",
    "Tech - AMZN": "AMZN", "Tech - META": "META", "Tech - GOOGL": "GOOGL", "Tech - NFLX": "NFLX",
    "Tech - SPCX": "SPCX",

    # AI Chips (6)
    "Chips - AMD": "AMD", "Chips - MU": "MU", "Chips - SNDK": "SNDK", "Chips - AVGO": "AVGO",
    "Chips - INTC": "INTC", "Chips - ARM": "ARM",

    # Commodities (4)
    "Comm - GOLD": "GC=F", "Comm - SILVER": "SI=F", "Comm - WTI": "CL=F", "Comm - COPPER": "HG=F"
}

# Assets where the Yahoo ticker may not match what SafeBets quotes.
# SPCX in particular: SpaceX is not publicly traded, so SafeBets is quoting
# some private/synthetic mark that Yahoo cannot possibly match. Any row that
# fails validation here should be entered by hand from the SafeBets tile.
UNVERIFIED_TICKERS = {"SPCX", "HYPE32196-USD", "SNDK"}

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


SELECTION_MARGIN = 0.005   # must beat the baseline by 0.5% relative to be chosen


def walk_forward_compare(X, y, days, n_folds):
    """
    Score three candidates out-of-sample on identical folds, then let the data
    pick one. The candidates sit at increasing levels of ambition:

      baseline : submit today's price (predict a log return of zero)
      drift    : submit today's price nudged by the median historical h-day
                 log return -- a single parameter, estimated on train only
      model    : the full 16-feature median ensemble

    The drift candidate matters because of a specific asymmetry. If price is
    roughly a martingale, its log is not: the median of the price distribution
    sits below the mean by about sigma^2/2. Under percentage-error scoring the
    median is what you want, so a small constant offset can beat a flat guess
    with a fraction of the estimation variance an ensemble carries. It was
    never tested on its own before -- shrinkage multiplied the ensemble's
    intercept away along with everything else.

    The label at row t is built from prices at t+days, so any test row within
    `days` of the training cut is contaminated. That gap is purged.
    """
    n = len(X)
    blank = {'best': 'baseline', 'skill': 0.0, 'drift': 0.0,
             'mae': {}, 'n_tested': 0}

    if n_folds < 1 or n < 250:
        return blank

    fold_size = n // (n_folds + 1)
    if fold_size <= days + 30:
        return blank

    preds_model, preds_drift, actuals = [], [], []

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + days
        test_end = min(test_start + fold_size, n)

        if train_end < 120 or test_end - test_start < 20:
            continue

        y_tr = y.iloc[:train_end]
        n_test = test_end - test_start

        try:
            predictor = fit_median_ensemble(
                X.iloc[:train_end], y_tr, _sample_weights(train_end)
            )
            preds_model.append(predictor(X.iloc[test_start:test_end]))
        except Exception:
            preds_model.append(np.zeros(n_test))

        # Drift estimated on training data only -- no peeking.
        preds_drift.append(np.full(n_test, float(np.median(y_tr))))
        actuals.append(y.iloc[test_start:test_end].to_numpy())

    if not actuals:
        return blank

    a = np.concatenate(actuals)
    pm = np.concatenate(preds_model)
    pd_ = np.concatenate(preds_drift)

    mae = {
        'baseline': float(np.mean(np.abs(a))),
        'drift': float(np.mean(np.abs(pd_ - a))),
        'model': float(np.mean(np.abs(pm - a))),
    }

    base = mae['baseline']
    if base <= 0:
        return blank

    # Pick the winner, but only if it clears the baseline by a real margin.
    best, best_mae = 'baseline', base
    for cand in ('drift', 'model'):
        if mae[cand] < best_mae and mae[cand] < base * (1 - SELECTION_MARGIN):
            best, best_mae = cand, mae[cand]

    return {
        'best': best,
        'skill': 1.0 - (best_mae / base),
        'drift': float(np.median(y)),
        'mae': mae,
        'n_tested': int(len(a)),
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_live_price(ticker_symbol):
    """
    Fetch the freshest price available, with fallbacks.

    This matters more than any model here. Every candidate is anchored on
    "today's price", and the daily close from yfinance can be hours stale for
    a 24/7 asset. Anchoring on a stale close adds the market's drift since
    that close to every submission, on top of the irreducible error -- and
    that gap is often larger than any edge the model was competing for.

    Returns (price, source) or (None, reason).
    """
    try:
        info = yf.Ticker(ticker_symbol).fast_info
        price = info.get("last_price") if hasattr(info, "get") else getattr(info, "last_price", None)
        if price and float(price) > 0:
            return float(price), "live"
    except Exception:
        pass

    try:
        intraday = yf.Ticker(ticker_symbol).history(period="1d", interval="1m")
        if not intraday.empty:
            return float(intraday['Close'].iloc[-1]), "1m bar"
    except Exception:
        pass

    return None, "stale close"


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

            cmp = walk_forward_compare(X, y, days, folds)
            horizon_sigma = daily_sigma * np.sqrt(days)

            if cmp['best'] == 'model':
                predictor = fit_median_ensemble(X, y, _sample_weights(len(X)))
                pred_log = float(predictor(X_today)[0])
            elif cmp['best'] == 'drift':
                pred_log = cmp['drift']
            else:
                pred_log = 0.0

            cap = MAX_SIGMA_CLIP * horizon_sigma
            pred_log = float(np.clip(pred_log, -cap, cap))

            shrink = float(np.clip(cmp['skill'], 0.0, 1.0)) if shrink_enabled else 1.0
            effective_log = pred_log * shrink

            results[name] = {
                'price': latest_price * float(np.exp(effective_log)),
                'log_offset': effective_log,
                'sigma': horizon_sigma,
                'pct': (float(np.exp(effective_log)) - 1.0) * 100.0,
                'source': cmp['best'],
                'skill': cmp['skill'],
                'mae': cmp['mae'],
                'n_tested': cmp['n_tested'],
                'lo': latest_price * float(np.exp(effective_log - BAND_Z * horizon_sigma)),
                'hi': latest_price * float(np.exp(effective_log + BAND_Z * horizon_sigma)),
            }

        if not results:
            return None, None

        return results, latest_price
    except Exception:
        return None, None



# --- 6. EXPECTED VALUE / ALLOCATION ENGINE ---
#
# The scoring is not continuous percentage error -- it is nested deviation
# bands with fixed payouts per timeframe. Bull's Eye probability turns out to
# be roughly constant across timeframes (the bands are volatility-calibrated),
# while payouts scale 50x from 24H to 30D. So which WINDOW you spend a
# prediction on matters far more than any forecasting edge.

PAYOUTS = {
    'HOURS_24': {'BULLS_EYE': 20,   'EXCELLENT': 10,  'GREAT': 5,   'GOOD': 1},
    'DAYS_7':   {'BULLS_EYE': 150,  'EXCELLENT': 50,  'GREAT': 20,  'GOOD': 10},
    'DAYS_14':  {'BULLS_EYE': 400,  'EXCELLENT': 150, 'GREAT': 50,  'GOOD': 20},
    'DAYS_30':  {'BULLS_EYE': 1000, 'EXCELLENT': 400, 'GREAT': 150, 'GOOD': 50},
}
PERIOD_DAYS = {'HOURS_24': 1, 'DAYS_7': 7, 'DAYS_14': 14, 'DAYS_30': 30}
TIER_ORDER = ['BULLS_EYE', 'EXCELLENT', 'GREAT', 'GOOD']

# Paste each symbol's /api/.../accuracy-thresholds response here to include it.
# Only symbols present below appear in the EV table.
THRESHOLDS = {
    "AAPL": [
        {"tier": "BULLS_EYE", "deviation": 0.05, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.3, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.35, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.7, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.5, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.7, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 2.5, "periodName": "DAYS_30"},
    ],
    "AMD": [
        {"tier": "BULLS_EYE", "deviation": 0.18, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.42, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.45, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.38, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.9, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.95, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.3, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.52, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.75, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.65, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.9, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.0, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.8, "periodName": "DAYS_30"},
    ],
    "AMZN": [
        {"tier": "BULLS_EYE", "deviation": 0.05, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.85, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 1.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_30"},
    ],
    "ARM": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.48, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.8, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.05, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.3, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.9, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.45, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.25, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.5, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.5, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.65, "periodName": "DAYS_30"},
    ],
    "AVAX": [
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 2.25, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 3.5, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 3.25, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 5.75, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 9.25, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 2.0, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 4.5, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 8.0, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 13.0, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 6.75, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 12.0, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 19.0, "periodName": "DAYS_30"},
    ],
    "AVGO": [
        {"tier": "BULLS_EYE", "deviation": 0.08, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.85, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.48, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.8, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.28, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.68, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.5, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.55, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.44, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.0, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 3.6, "periodName": "DAYS_30"},
    ],
    "BTC": [
        {"tier": "BULLS_EYE", "deviation": 0.34, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 0.69, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 1.43, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 2.54, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.7, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.15, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.6, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.1, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.85, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.9, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.35, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.75, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.35, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.4, "periodName": "DAYS_30"},
    ],
    "COPPER": [
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.65, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.1, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.8, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.6, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.5, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 4.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.9, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.4, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.5, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.2, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 5.1, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 8.2, "periodName": "DAYS_30"},
    ],
    "DOGE": [
        {"tier": "BULLS_EYE", "deviation": 0.62, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 2.59, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 4.61, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.85, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.35, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.5, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.75, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.1, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.15, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.55, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.25, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.65, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 3.3, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 5.2, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.6, "periodName": "DAYS_30"},
    ],
    "ETH": [
        {"tier": "BULLS_EYE", "deviation": 0.38, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 0.77, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 1.59, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 2.84, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.45, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.75, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.75, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 4.2, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.15, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.35, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.85, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.65, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 3.4, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 5.6, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 8.5, "periodName": "DAYS_30"},
    ],
    "GOLD": [
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.75, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.3, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.05, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.85, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.75, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.5, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.7, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.2, "periodName": "DAYS_30"},
    ],
    "GOOGL": [
        {"tier": "BULLS_EYE", "deviation": 0.05, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.8, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.15, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 1.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_30"},
    ],
    "HYPE": [
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.75, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.8, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 3.05, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 4.7, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.6, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 4.35, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 6.7, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.0, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 4.0, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 6.65, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 9.75, "periodName": "DAYS_30"},
    ],
    "INTC": [
        {"tier": "BULLS_EYE", "deviation": 0.16, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.85, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.4, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.88, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.85, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.15, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.65, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.45, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.75, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.75, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 3.75, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.3, "periodName": "DAYS_30"},
    ],
    "LINK": [
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 1.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 2.1, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 3.3, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 3.0, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 5.5, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 8.75, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.85, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 4.3, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 7.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 12.3, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.7, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 6.25, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 11.25, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 18.0, "periodName": "DAYS_30"},
    ],
    "META": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.75, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.8, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.1, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.4, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.05, "periodName": "DAYS_30"},
    ],
    "MSFT": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.7, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.9, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.55, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.3, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.2, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.4, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.9, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 3.2, "periodName": "DAYS_30"},
    ],
    "MU": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.95, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.4, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.95, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.05, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.5, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.55, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.35, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.9, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.95, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.15, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.0, "periodName": "DAYS_30"},
    ],
    "NFLX": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.75, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.5, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.7, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.7, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.45, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.45, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.4, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.95, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 3.55, "periodName": "DAYS_30"},
    ],
    "NVDA": [
        {"tier": "BULLS_EYE", "deviation": 0.12, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.6, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.85, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.15, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.1, "periodName": "DAYS_30"},
    ],
    "SILVER": [
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.35, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.35, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.9, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.1, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.2, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.5, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.1, "periodName": "DAYS_30"},
    ],
    "SNDK": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.48, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.75, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.05, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.25, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.8, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.45, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.15, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.35, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.05, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.5, "periodName": "DAYS_30"},
    ],
    "SOL": [
        {"tier": "BULLS_EYE", "deviation": 0.43, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 0.86, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 1.78, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 3.18, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.8, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.25, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.7, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.3, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.55, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.0, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.0, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.35, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.45, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.95, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.8, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.1, "periodName": "DAYS_30"},
    ],
    "SPCX": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.1, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.25, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.6, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.6, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.15, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.0, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.25, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.5, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.25, "periodName": "DAYS_30"},
    ],
    "TSLA": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.8, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.15, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.1, "periodName": "DAYS_30"},
    ],
    "WTI": [
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 1.1, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.9, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 3.0, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 1.0, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 2.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 4.25, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 6.8, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.4, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 3.3, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 5.8, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 9.3, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 4.9, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 8.75, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 14.0, "periodName": "DAYS_30"},
    ],
}

# app ticker -> SafeBets symbol
SYMBOL_MAP = {v: k.split(" - ")[-1] for k, v in asset_map.items()}


@st.cache_data(ttl=3600, show_spinner=False)
def empirical_deviations(ticker_symbol, days):
    """
    Actual historical |percentage move| over `days`, from this asset's own
    history. Used instead of a normal approximation because return
    distributions are leptokurtic: more mass near zero AND fatter tails than
    a Gaussian, which shifts tight-band hit rates meaningfully.
    """
    try:
        raw = yf.Ticker(ticker_symbol).history(period=HISTORY_PERIOD)
        if raw.empty or len(raw) < days + 60:
            return None
        close = raw['Close']
        moves = (close.shift(-days) / close - 1.0).dropna().abs() * 100.0
        return moves.to_numpy()
    except Exception:
        return None


def ev_for_symbol(ticker_symbol, sb_symbol):
    """Expected unicoins per prediction, per timeframe, using empirical bands."""
    bands = THRESHOLDS.get(sb_symbol)
    if not bands:
        return None

    by_period = {}
    for row in bands:
        by_period.setdefault(row['periodName'], []).append(
            (float(row['deviation']), row['tier'])
        )

    out = {}
    for period, days in PERIOD_DAYS.items():
        if period not in by_period or period not in PAYOUTS:
            continue
        moves = empirical_deviations(ticker_symbol, days)
        if moves is None or len(moves) < 100:
            continue

        ordered = sorted(by_period[period])
        ev, prev_cum, tier_probs = 0.0, 0.0, {}
        for dev, tier in ordered:
            cum = float(np.mean(moves <= dev))
            excl = max(0.0, cum - prev_cum)
            ev += excl * PAYOUTS[period].get(tier, 0)
            tier_probs[tier] = excl
            prev_cum = cum

        out[period] = {
            'ev': ev,
            'p_any': prev_cum,
            'probs': tier_probs,
            'n': len(moves),
        }
    return out


# --- 7. CONDITIONAL VOLATILITY ENGINE ---
#
# Under band scoring, P(hit) is roughly band_width x density_at_prediction,
# which scales as 1/sigma. Direction is unforecastable (the backtest above
# settled that), but volatility is not -- it clusters. So the edge is not
# predicting where price goes; it is predicting how far it travels, and
# spending predictions only when the answer is "not far".

VOL_REVERSION_DAYS = 20.0


def forecast_vol(log_ret, days):
    """
    Forecast the h-day return sigma, as a series aligned to each date.

    Blends a fast EWMA estimate of current conditions against the long-run
    level, weighting toward the long run as the horizon extends, because
    volatility mean-reverts. Uses only trailing data at every point, so it is
    safe to evaluate out-of-sample.
    """
    short = log_ret.ewm(halflife=10, min_periods=20).std()
    long = log_ret.rolling(252, min_periods=60).std()
    w = float(np.exp(-days / VOL_REVERSION_DAYS))
    daily = np.sqrt(w * short**2 + (1 - w) * long**2)
    return daily * np.sqrt(days)


@st.cache_data(ttl=1800, show_spinner=False)
def conditional_vol_state(ticker_symbol):
    """
    Returns today's forecast sigma per horizon, its historical percentile, and
    an out-of-sample check that low-forecast-vol days really do hit more often.
    """
    try:
        raw = yf.Ticker(ticker_symbol).history(period=HISTORY_PERIOD)
        if raw.empty or len(raw) < 400:
            return None

        close = raw['Close']
        log_ret = np.log(close).diff()
        out = {}

        for period, days in PERIOD_DAYS.items():
            fvol = forecast_vol(log_ret, days)
            realised = (np.log(close).shift(-days) - np.log(close)).abs()

            frame = pd.DataFrame({'fvol': fvol, 'realised': realised}).dropna()
            if len(frame) < 200:
                continue

            today_vol = float(fvol.dropna().iloc[-1])
            pct = float((fvol.dropna() < today_vol).mean())

            # Standardised residuals: divide each realised move by the vol that
            # was forecast for it. Rescaling these by today's forecast keeps the
            # fat tails while conditioning on the current regime.
            z = (frame['realised'] / frame['fvol']).to_numpy()

            # Out-of-sample check: does a low forecast actually predict a small
            # move? Compare realised moves in the calmest vs busiest tercile.
            q_lo, q_hi = frame['fvol'].quantile([0.33, 0.67])
            calm = frame.loc[frame['fvol'] <= q_lo, 'realised'].mean()
            rough = frame.loc[frame['fvol'] >= q_hi, 'realised'].mean()
            ratio = float(rough / calm) if calm > 0 else np.nan

            out[period] = {
                'sigma': today_vol,
                'percentile': pct,
                'z': z,
                'calm_move': float(calm) * 100,
                'rough_move': float(rough) * 100,
                'ratio': ratio,
                'n': len(frame),
            }
        return out
    except Exception:
        return None


def conditional_ev(ticker_symbol, sb_symbol):
    """Expected unicoins per prediction given TODAY's volatility regime."""
    bands = THRESHOLDS.get(sb_symbol)
    state = conditional_vol_state(ticker_symbol)
    if not bands or not state:
        return None

    by_period = {}
    for row in bands:
        by_period.setdefault(row['periodName'], []).append(
            (float(row['deviation']), row['tier'])
        )

    out = {}
    for period, days in PERIOD_DAYS.items():
        if period not in by_period or period not in state or period not in PAYOUTS:
            continue

        st_ = state[period]
        # Rescale the standardised move distribution to today's forecast vol.
        moves = st_['z'] * st_['sigma'] * 100.0

        ordered = sorted(by_period[period])
        ev, prev, probs = 0.0, 0.0, {}
        for dev, tier in ordered:
            cum = float(np.mean(moves <= dev))
            excl = max(0.0, cum - prev)
            ev += excl * PAYOUTS[period].get(tier, 0)
            probs[tier] = excl
            prev = cum

        out[period] = {
            'ev': ev,
            'p_any': prev,
            'probs': probs,
            'vol_pct': st_['percentile'],
            'sigma': st_['sigma'] * 100,
            'ratio': st_['ratio'],
            'n': st_['n'],
        }
    return out

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

        results, close_price = get_predictions(ticker, apply_shrinkage, n_folds)
        sent_score, sent_status = sentiments.get(asset_name, (0.0, "n/a"))

        if results:
            live, live_src = get_live_price(ticker)
            price = live if (use_live and live) else close_price
            gap_pct = ((live / close_price - 1.0) * 100.0) if live else 0.0

            fmt_p = "${:,.4f}" if price < 1 else "${:,.2f}"
            sub = {"Asset": asset_name, "Anchor": fmt_p.format(price)}
            if ticker in UNVERIFIED_TICKERS:
                sub["Anchor"] = "⚠️ " + sub["Anchor"]
            if use_live:
                sub["Staleness"] = (
                    f"{gap_pct:+.2f}% ({live_src})" if live else "— stale close"
                )
            diag = {"Asset": asset_name}

            for h in HORIZONS:
                if h not in results:
                    sub[h] = "n/a"
                    diag[f"{h} skill"] = "n/a"
                    continue

                r = results[h]
                # Re-anchor on the live price: the model contributes a log
                # offset, the anchor supplies the level.
                final = price * float(np.exp(r['log_offset']))
                final *= (1 + sent_score * sentiment_weight)
                pct = (final / price - 1.0) * 100.0

                cell = f"${final:,.4f}" if final < 1 else f"${final:,.2f}"
                cell += f"  ({pct:+.2f}%)"
                if show_bands:
                    lo = final * float(np.exp(-BAND_Z * r['sigma']))
                    hi = final * float(np.exp(BAND_Z * r['sigma']))
                    fmt = "{:,.4f}" if hi < 1 else "{:,.2f}"
                    cell += f"\n[{fmt.format(lo)} – {fmt.format(hi)}]"
                sub[h] = cell

                icon = {'baseline': '⚪ today', 'drift': '🟡 drift', 'model': '🟢 model'}
                diag[f"{h} uses"] = icon.get(r['source'], r['source'])
                m = r['mae']
                diag[f"{h} err"] = (
                    f"today {m['baseline']*100:.2f}% | "
                    f"drift {m['drift']*100:.2f}% | "
                    f"model {m['model']*100:.2f}%"
                ) if m else "n/a"

            submission_rows.append(sub)
            diagnostic_rows.append(diag)

        progress_bar.progress((idx + 1) / total)

    status_text.text("Analysis complete!")

    if submission_rows:
        st.subheader("Values to submit")
        st.dataframe(pd.DataFrame(submission_rows), use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ marks a ticker where the Yahoo symbol may not be the instrument "
            "SafeBets quotes — check that anchor against the platform tile before "
            "submitting, or type the tile price in by hand."
        )

        st.subheader("Did the model beat 'just enter today's price'?")
        st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)

        st.markdown(
            "**Uses** shows which of three candidates won on out-of-sample data "
            "for that asset and horizon: `today` (submit the current price), "
            "`drift` (current price nudged by the median historical log return, "
            "one parameter), or `model` (the full ensemble). **Err** lists all "
            "three mean absolute percentage errors so you can see the margin. "
            "A candidate only displaces `today` if it beats it by at least "
            "0.5% relative, so ties go to the simpler option.\n\n"
            "Most cells reading `today` is the expected outcome, not a broken "
            "model — under percentage-error scoring the current price is a very "
            "strong submission. If `drift` wins on the longer horizons, that is "
            "the volatility-drag effect being picked up, and it is the most "
            "likely place to find a real edge."
        )

st.markdown("---")
st.subheader("Where to spend your predictions")
st.caption(
    "Expected unicoins per prediction, using each asset's own historical move "
    "distribution against the platform's deviation bands. Only symbols with "
    "thresholds pasted into THRESHOLDS appear here."
)

if st.button("💰 Rank timeframes by expected value"):
    ev_rows = []
    bar = st.progress(0)
    covered = [(n, t) for n, t in asset_map.items() if SYMBOL_MAP.get(t) in THRESHOLDS]

    if not covered:
        st.warning("No symbols have thresholds configured. Paste accuracy-thresholds JSON into THRESHOLDS.")
    else:
        for i, (asset_name, ticker) in enumerate(covered):
            res = ev_for_symbol(ticker, SYMBOL_MAP.get(ticker))
            if res:
                for period, d in res.items():
                    ev_rows.append({
                        "Asset": asset_name,
                        "Timeframe": period,
                        "EV (ú)": round(d['ev'], 1),
                        "P(any tier)": f"{d['p_any']:.1%}",
                        "P(Bull's Eye)": f"{d['probs'].get('BULLS_EYE', 0):.1%}",
                        "Samples": d['n'],
                    })
            bar.progress((i + 1) / len(covered))

        if ev_rows:
            ev_df = pd.DataFrame(ev_rows).sort_values("EV (ú)", ascending=False)
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

            best = ev_df.iloc[0]
            worst = ev_df.iloc[-1]
            ratio = best["EV (ú)"] / worst["EV (ú)"] if worst["EV (ú)"] > 0 else float('inf')
            st.info(
                f"Best: **{best['Asset']} / {best['Timeframe']}** at {best['EV (ú)']} ú per "
                f"prediction. Worst: **{worst['Asset']} / {worst['Timeframe']}** at "
                f"{worst['EV (ú)']} ú — a {ratio:.0f}x difference for the same one-unicoin "
                "stake. Spend your daily windows top-down this list."
            )
            st.caption(
                "Probabilities come from overlapping historical windows, so samples "
                "are autocorrelated: treat these as well-grounded estimates, not "
                "precise odds. They also assume you submit the current price, which "
                "the backtest above showed beats every model tested."
            )

st.markdown("---")
st.subheader("Submit today, or wait?")
st.caption(
    "Hit probability scales as 1/sigma, so the same prediction is worth two to "
    "three times as much in calm conditions as in turbulent ones. This ranks "
    "every asset and timeframe by expected value GIVEN today's volatility."
)

if st.button("🎯 Rank by today's conditions"):
    rows = []
    bar = st.progress(0)
    covered = [(n, t) for n, t in asset_map.items() if SYMBOL_MAP.get(t) in THRESHOLDS]

    if not covered:
        st.warning(
            "No symbols configured. Paste each asset's accuracy-thresholds "
            "JSON into the THRESHOLDS dict near the top of this file."
        )
    else:
        for i, (asset_name, ticker) in enumerate(covered):
            res = conditional_ev(ticker, SYMBOL_MAP.get(ticker))
            if res:
                for period, d in res.items():
                    if d['vol_pct'] < 0.33:
                        signal = "🟢 calm — submit"
                    elif d['vol_pct'] < 0.67:
                        signal = "⚪ normal"
                    else:
                        signal = "🔴 turbulent — wait"
                    rows.append({
                        "Asset": asset_name,
                        "Timeframe": period,
                        "EV now (ú)": round(d['ev'], 1),
                        "Signal": signal,
                        "Vol pct": f"{d['vol_pct']:.0%}",
                        "Fcst σ": f"{d['sigma']:.2f}%",
                        "P(any tier)": f"{d['p_any']:.1%}",
                        "P(Bull's Eye)": f"{d['probs'].get('BULLS_EYE', 0):.1%}",
                    })
            bar.progress((i + 1) / len(covered))

        if rows:
            df = pd.DataFrame(rows).sort_values("EV now (ú)", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

            calm = df[df["Signal"].str.contains("calm")]
            st.info(
                f"**{len(calm)} of {len(df)}** asset/timeframe slots are in the calmest "
                "third of their own history right now. Those are where a prediction is "
                "worth most today. Slots marked turbulent will be worth more later — "
                "the window reopens daily, so skipping one costs nothing."
            )

            # Does the vol forecast work? Checked at EVERY horizon, not just 30D:
            # persistence is strong over days and washes out over weeks, so the
            # answer can differ completely across the four windows.
            check_rows, ratios_by_period = [], {p: [] for p in PERIOD_DAYS}
            for asset_name, ticker in covered:
                stt = conditional_vol_state(ticker)
                if not stt:
                    continue
                row = {"Asset": asset_name}
                for period in ['HOURS_24', 'DAYS_7', 'DAYS_14', 'DAYS_30']:
                    if period in stt and np.isfinite(stt[period]['ratio']):
                        r = stt[period]['ratio']
                        ratios_by_period[period].append(r)
                        flag = "✓" if r >= 1.15 else ("~" if r >= 1.05 else "✗")
                        row[period] = f"{flag} {r:.2f}x"
                    else:
                        row[period] = "n/a"
                check_rows.append(row)

            if check_rows:
                st.markdown(
                    "**Does the volatility forecast actually work?** Ratio of the "
                    "average realised move on turbulent days versus calm days, out of "
                    "sample, per horizon. Above 1.0 means the forecast separates the "
                    "two and the timing signal is real."
                )
                st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

                verdicts = []
                for period in ['HOURS_24', 'DAYS_7', 'DAYS_14', 'DAYS_30']:
                    vals = ratios_by_period[period]
                    if not vals:
                        continue
                    med = float(np.median(vals))
                    if med >= 1.15:
                        verdicts.append(f"**{period}: use it** (median {med:.2f}x)")
                    elif med >= 1.05:
                        verdicts.append(f"{period}: marginal ({med:.2f}x)")
                    else:
                        verdicts.append(f"{period}: **ignore the signal** ({med:.2f}x)")
                st.markdown(
                    "Verdict per horizon — " + " · ".join(verdicts) +
                    "\n\nWhere the verdict is *ignore*, disregard the calm/turbulent "
                    "column above and use the unconditional ranking instead. A ratio "
                    "at or below 1.0 means the forecast has no skill at that horizon, "
                    "so skipping a slot on its say-so costs you expected value for "
                    "nothing. ✓ = 1.15x or better, ~ = 1.05-1.15x, ✗ = below 1.05x."
                )