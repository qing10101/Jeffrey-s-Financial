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
    volume = raw['Volume'][SYMBOL].ffill()
    closes.columns = ['BTC', 'NASDAQ', 'VIX']

    df = pd.DataFrame(index=closes.index)
    df['btc'] = closes['BTC']
    df['nasdaq'] = closes['NASDAQ']
    df['vix'] = closes['VIX']

    # --- ALPHA FEATURES (WITH INFINITY PROTECTION) ---
    df['ret'] = df['btc'].pct_change()

    # 1. Kurtosis Proxy (Add epsilon 1e-9 to prevent div by zero)
    vol_20 = df['ret'].rolling(20).std()
    df['kurt_proxy'] = df['ret'].rolling(5).max() / (vol_20 + 1e-9)

    # 2. Velocity (Using diff instead of pct_change to avoid Infinity)
    # This measures if momentum is accelerating or decelerating
    df['velocity'] = df['ret'].diff(3)

    # 3. Market Regime
    df['sma_200'] = df['btc'].rolling(200).mean()
    df['dist_sma200'] = (df['btc'] - df['sma_200']) / (df['sma_200'] + 1e-9)

    # 4. Nasdaq Liquidity
    df['nasdaq_ret_10'] = df['nasdaq'].pct_change(10)

    # Targets
    df['target'] = (df['btc'].shift(-3) > df['btc']).astype(int)
    df['next_ret'] = df['ret'].shift(-1)

    # --- CLEANING STEP ---
    # Replace Infinity with NaN, then drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()


df = get_btc_alpha_fixed_data()
features = ['kurt_proxy', 'velocity', 'dist_sma200', 'nasdaq_ret_10', 'vix']

# --- WALK-FORWARD LOOP ---
results_list = []
start_idx = TRAIN_WINDOW

print(f"Executing Alpha-Max Backtest...")
for i in range(start_idx, len(df) - STEP_SIZE, STEP_SIZE):
    train_df = df.iloc[i - TRAIN_WINDOW:i]
    test_df = df.iloc[i:i + STEP_SIZE]

    # Train
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train_df[features], train_df['target'])

    probs = model.predict_proba(test_df[features])[:, 1]

    batch = pd.DataFrame({
        'date': test_df.index,
        'actual_ret': test_df['next_ret'],
        'prob': probs,
        'dist_sma200': test_df['dist_sma200']
    }, index=test_df.index)
    results_list.append(batch)

results = pd.concat(results_list)

# --- DYNAMIC LEVERAGE LOGIC ---
positions = []
current_pos = 0

for i in range(len(results)):
    p = results['prob'].iloc[i]
    d = results['dist_sma200'].iloc[i]

    # Bull Run: Leverage 1.5x
    if d > 0.1:
        buy_t, sell_t, mult = 0.46, 0.40, 1.5
    # Neutral/Recovery: Standard 1.0x
    elif d > -0.1:
        buy_t, sell_t, mult = 0.52, 0.48, 1.0
    # Bear: De-risk to 0.5x
    else:
        buy_t, sell_t, mult = 0.58, 0.52, 0.5

    if p > buy_t:
        current_pos = mult
    elif p < sell_t:
        current_pos = 0
    positions.append(current_pos)

results['position'] = positions
results['cost'] = (results['position'].diff().abs().fillna(0) * TRANSACTION_COST)
results['strat_ret'] = (results['position'] * results['actual_ret']) - results['cost']

results['cum_market'] = (1 + results['actual_ret']).cumprod()
results['cum_strat'] = (1 + results['strat_ret']).cumprod()

# --- EVALUATION ---
sharpe = (results['strat_ret'].mean() / results['strat_ret'].std()) * np.sqrt(365)
print("\n" + "=" * 30)
print(f"BITCOIN ALPHA-MAX FIXED")
print("=" * 30)
print(f"Strategy Sharpe: {sharpe:.2f}")
print(f"Strategy Ret:    {(results['cum_strat'].iloc[-1] - 1):.2%}")
print(f"Market Ret:      {(results['cum_market'].iloc[-1] - 1):.2%}")

plt.figure(figsize=(12, 6))
plt.plot(results['date'], results['cum_market'], label='BTC HODL', color='orange', alpha=0.3)
plt.plot(results['date'], results['cum_strat'], label='Alpha-Max Strategy', color='red', linewidth=2)
plt.yscale('log')
plt.title("Bitcoin Alpha-Max: Dynamic Leverage (Fixed)")
plt.legend()
plt.show()