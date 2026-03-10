import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
SYMBOL = "BTC-USD"
MACRO_TICKERS = ["^IXIC", "^VIX"]
TRANSACTION_COST = 0.001
TRAIN_WINDOW = 1200
STEP_SIZE = 14


def get_btc_alpha_fixed_data():
    print(f"Fetching BTC Data and cleaning Infinities...")
    raw = yf.download([SYMBOL] + MACRO_TICKERS, period="max", auto_adjust=False)
    closes = raw['Close'].ffill()

    # Handle yfinance MultiIndex
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = closes.columns.get_level_values(1)

    mapping = {SYMBOL: 'BTC', '^IXIC': 'NASDAQ', '^VIX': 'VIX'}
    closes = closes.rename(columns=mapping)

    df = pd.DataFrame(index=closes.index)
    df['btc'] = closes['BTC']
    df['nasdaq'] = closes['NASDAQ']
    df['vix'] = closes['VIX']

    df['ret'] = df['btc'].pct_change()
    vol_20 = df['ret'].rolling(20).std()
    df['kurt_proxy'] = df['ret'].rolling(5).max() / (vol_20 + 1e-9)
    df['velocity'] = df['ret'].diff(3)
    df['sma_200'] = df['btc'].rolling(200).mean()
    df['dist_sma200'] = (df['btc'] - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['nasdaq_ret_10'] = df['nasdaq'].pct_change(10)
    df['target'] = (df['btc'].shift(-3) > df['btc']).astype(int)
    df['next_ret'] = df['ret'].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


df = get_btc_alpha_fixed_data()
features = ['kurt_proxy', 'velocity', 'dist_sma200', 'nasdaq_ret_10', 'vix']

# --- WALK-FORWARD LOOP ---
results_list = []
start_idx = TRAIN_WINDOW
last_model = None

print(f"Executing Alpha-Max Backtest...")
for i in range(start_idx, len(df) - STEP_SIZE, STEP_SIZE):
    train_df = df.iloc[i - TRAIN_WINDOW:i]
    test_df = df.iloc[i:i + STEP_SIZE]

    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train_df[features], train_df['target'])
    last_model = model

    probs = model.predict_proba(test_df[features])[:, 1]
    batch = pd.DataFrame({
        'date': test_df.index,
        'actual_ret': test_df['next_ret'],
        'prob': probs,
        'dist_sma200': test_df['dist_sma200'],
        'target': test_df['target']
    }, index=test_df.index)
    results_list.append(batch)

results = pd.concat(results_list)

# --- DYNAMIC LEVERAGE LOGIC ---
positions = []
current_pos = 0
for i in range(len(results)):
    p, d = results['prob'].iloc[i], results['dist_sma200'].iloc[i]
    if d > 0.1:
        buy_t, sell_t, mult = 0.46, 0.40, 1.5
    elif d > -0.1:
        buy_t, sell_t, mult = 0.52, 0.48, 1.0
    else:
        buy_t, sell_t, mult = 0.58, 0.52, 0.5

    if p > buy_t:
        current_pos = mult
    elif p < sell_t:
        current_pos = 0
    positions.append(current_pos)

results['position'] = positions
results['strat_ret'] = (results['position'] * results['actual_ret']) - (
            results['position'].diff().abs().fillna(0) * TRANSACTION_COST)
results['cum_market'] = (1 + results['actual_ret']).cumprod()
results['cum_strat'] = (1 + results['strat_ret']).cumprod()

# --- EVALUATION ---
print("\n" + "=" * 30)
print(f"BITCOIN ALPHA-MAX RESULTS")
print("=" * 30)
print(f"Strategy Sharpe: {(results['strat_ret'].mean() / results['strat_ret'].std()) * np.sqrt(365):.2f}")
print(f"Strategy Ret:    {(results['cum_strat'].iloc[-1] - 1):.2%}")
print(f"Market Ret:      {(results['cum_market'].iloc[-1] - 1):.2%}")

# --- PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(results['date'], results['cum_market'], label='BTC HODL', color='orange', alpha=0.3)
plt.plot(results['date'], results['cum_strat'], label='Alpha-Max Strategy', color='red', linewidth=2)
plt.yscale('log')
plt.title("Bitcoin Alpha-Max: Dynamic Leverage (Fixed)")
plt.legend()
plt.grid(True, alpha=0.3)


# --- LIVE SIGNAL ---
def get_live_btc_alpha_signal(model):
    print("\n" + "=" * 40)
    print("   BITCOIN ALPHA-MAX: LIVE SIGNAL")
    print("=" * 40)
    raw = yf.download([SYMBOL] + MACRO_TICKERS, period="300d", auto_adjust=False)['Close'].ffill()
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(1)
    mapping = {SYMBOL: 'BTC', '^IXIC': 'NASDAQ', '^VIX': 'VIX'}
    raw = raw.rename(columns=mapping)

    ret = raw['BTC'].pct_change()
    vol_20 = ret.rolling(20).std().iloc[-1]
    kurt = ret.rolling(5).max().iloc[-1] / (vol_20 + 1e-9)
    velocity = ret.diff(3).iloc[-1]
    sma_200 = raw['BTC'].rolling(200).mean().iloc[-1]
    dist_200 = (raw['BTC'].iloc[-1] - sma_200) / (sma_200 + 1e-9)

    row = {'kurt_proxy': kurt, 'velocity': velocity, 'dist_sma200': dist_200,
           'nasdaq_ret_10': raw['NASDAQ'].pct_change(10).iloc[-1], 'vix': raw['VIX'].iloc[-1]}

    prob = model.predict_proba(pd.DataFrame([row])[features])[0, 1]
    label = "BULL" if dist_200 > 0.1 else ("BEAR" if dist_200 < -0.1 else "NEUTRAL")

    print(f"Date: {raw.index[-1].date()} | Regime: {label} | Prob: {prob:.2%}")
    if prob > 0.50:
        print(f"SUGGESTION: LONG ({label} POSITION SIZING)")
    else:
        print("SUGGESTION: CASH")


if last_model:
    get_live_btc_alpha_signal(last_model)
plt.show()