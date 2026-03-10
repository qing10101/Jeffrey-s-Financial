import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
SYMBOL = "GC=F"
MACRO_TICKERS = ["TIP", "DX-Y.NYB", "^GSPC", "^VIX"]
TRANSACTION_COST = 0.0001
TRAIN_WINDOW = 1000
STEP_SIZE = 20


def get_gold_recovery_data():
    print(f"Fetching Gold Macro-Basics...")
    raw = yf.download([SYMBOL] + MACRO_TICKERS, period="max", auto_adjust=False)
    closes = raw['Close'].ffill()
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = closes.columns.get_level_values(1)

    mapping = {SYMBOL: 'GOLD', 'TIP': 'TIP', 'DX-Y.NYB': 'DXY', '^GSPC': 'SPY'}
    closes = closes.rename(columns=mapping)

    df = pd.DataFrame(index=closes.index)
    df['gold'], df['tip'], df['dxy'], df['spy'] = closes['GOLD'], closes['TIP'], closes['DXY'], closes['SPY']
    df['ret'] = df['gold'].pct_change()
    df['dxy_ret'], df['tip_ret'] = df['dxy'].pct_change(), df['tip'].pct_change()
    df['gold_vs_spy'] = df['ret'] - df['spy'].pct_change()
    df['sma_50'] = df['gold'].rolling(50).mean()
    df['dist_sma50'] = (df['gold'] - df['sma_50']) / df['sma_50']
    df['ret_lag_1'] = df['ret'].shift(1)
    df['target'] = (df['gold'].shift(-2) > df['gold']).astype(int)
    df['next_ret'] = df['ret'].shift(-1)
    return df.dropna()


df = get_gold_recovery_data()
features = ['dxy_ret', 'tip_ret', 'gold_vs_spy', 'dist_sma50', 'ret_lag_1']

# --- WALK-FORWARD LOOP ---
results_list = []
start_idx = TRAIN_WINDOW
last_model = None

print(f"Executing Gold Recovery Backtest...")
for i in range(start_idx, len(df) - STEP_SIZE, STEP_SIZE):
    train_df, test_df = df.iloc[i - TRAIN_WINDOW:i], df.iloc[i:i + STEP_SIZE]
    model = xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.01, random_state=42)
    model.fit(train_df[features], train_df['target'])
    last_model = model

    probs = model.predict_proba(test_df[features])[:, 1]
    batch = pd.DataFrame({
        'date': test_df.index,
        'actual_ret': test_df['next_ret'],
        'prob': probs,
        'dist_sma50': test_df['dist_sma50'],
        'target': test_df['target']
    }, index=test_df.index)
    results_list.append(batch)

results = pd.concat(results_list)

# --- POSITION LOGIC ---
positions, current_pos = [], 0
for i in range(len(results)):
    p, d = results['prob'].iloc[i], results['dist_sma50'].iloc[i]
    buy_t, sell_t = (0.49, 0.45) if d > 0.02 else (0.53, 0.49)
    if p > buy_t:
        current_pos = 1.0
    elif p < sell_t:
        current_pos = 0
    positions.append(current_pos)

results['position'] = positions
results['strat_ret'] = (results['position'] * results['actual_ret']) - (
            results['position'].diff().abs().fillna(0) * TRANSACTION_COST)
results['cum_market'], results['cum_strat'] = (1 + results['actual_ret']).cumprod(), (
            1 + results['strat_ret']).cumprod()

# --- EVALUATION ---
print("\n" + "=" * 30)
print(f"GOLD RECOVERY RESULTS")
print("=" * 30)
print(f"Strategy Sharpe: {(results['strat_ret'].mean() / results['strat_ret'].std()) * np.sqrt(252):.2f}")
print(f"Strategy Ret:    {(results['cum_strat'].iloc[-1] - 1):.2%}")
print(f"Market Ret:      {(results['cum_market'].iloc[-1] - 1):.2%}")

# --- PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(results['date'], results['cum_market'], label='Gold Spot', color='gold', alpha=0.3)
plt.plot(results['date'], results['cum_strat'], label='Recovery Strategy', color='darkgreen', linewidth=2)
plt.title("Gold Recovery: Back to Macro-Basics")
plt.legend()
plt.grid(True, alpha=0.3)


# --- LIVE SIGNAL ---
def get_live_gold_recovery_signal(model):
    print("\n" + "=" * 40)
    print("   GOLD RECOVERY: LIVE SIGNAL")
    print("=" * 40)
    raw = yf.download([SYMBOL, "DX-Y.NYB", "TIP", "^GSPC"], period="60d", auto_adjust=False)['Close'].ffill()
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(1)
    mapping = {SYMBOL: 'GOLD', 'DX-Y.NYB': 'DXY', 'TIP': 'TIP', '^GSPC': 'SPY'}
    raw = raw.rename(columns=mapping)

    dist_sma50 = (raw['GOLD'].iloc[-1] - raw['GOLD'].rolling(50).mean().iloc[-1]) / raw['GOLD'].rolling(50).mean().iloc[
        -1]
    row = {'dxy_ret': raw['DXY'].pct_change().iloc[-1], 'tip_ret': raw['TIP'].pct_change().iloc[-1],
           'gold_vs_spy': raw['GOLD'].pct_change().iloc[-1] - raw['SPY'].pct_change().iloc[-1],
           'dist_sma50': dist_sma50, 'ret_lag_1': raw['GOLD'].pct_change().iloc[-1]}

    prob = model.predict_proba(pd.DataFrame([row])[features])[0, 1]
    print(f"Date: {raw.index[-1].date()} | Prob: {prob:.2%}")
    if prob > 0.50:
        print("SUGGESTION: LONG")
    else:
        print("SUGGESTION: CASH")


if last_model:
    get_live_gold_recovery_signal(last_model)
plt.show()