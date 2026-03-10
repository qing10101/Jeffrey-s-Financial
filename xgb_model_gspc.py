import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
SYMBOL = "^GSPC"
MACRO_TICKERS = ["^TNX", "DX-Y.NYB", "^VIX"]
TRANSACTION_COST = 0.0001  # 1 Basis Point
TRAIN_WINDOW = 500
STEP_SIZE = 20


def get_data():
    print("Fetching data from Yahoo Finance...")
    all_tickers = [SYMBOL] + MACRO_TICKERS
    raw_data = yf.download(all_tickers, period="max", auto_adjust=False)

    closes = raw_data['Close'].ffill()
    closes.columns = ['DXY', 'GSPC', 'TNX', 'VIX']
    gspc_open = raw_data['Open'][SYMBOL].ffill()

    df = closes.copy()
    df['open'] = gspc_open
    df['ret'] = df['GSPC'].pct_change()
    df['vix_ret'] = df['VIX'].pct_change()
    df['tnx_ret'] = df['TNX'].pct_change()

    # Feature Engineering
    df['rsi'] = (df['GSPC'] > df['GSPC'].shift(1)).rolling(14).mean()
    df['overnight_gap'] = (df['open'] - df['GSPC'].shift(1)) / df['GSPC'].shift(1)
    df['sma_200'] = df['GSPC'].rolling(200).mean()
    df['dist_sma200'] = (df['GSPC'] - df['sma_200']) / df['sma_200']

    for i in range(1, 4):
        df[f'ret_lag_{i}'] = df['ret'].shift(i)

    df['target'] = (df['GSPC'].shift(-1) > df['GSPC']).astype(int)
    df['next_ret'] = df['ret'].shift(-1)

    return df.dropna()


df = get_data()
features = ['rsi', 'overnight_gap', 'dist_sma200', 'vix_ret', 'tnx_ret', 'ret_lag_1']

# --- WALK-FORWARD VALIDATION LOOP ---
results_list = []
start_idx = TRAIN_WINDOW
last_model = None

print(f"Starting Walk-Forward Optimization on {len(df)} rows...")
for i in range(start_idx, len(df) - STEP_SIZE, STEP_SIZE):
    train_df = df.iloc[i - TRAIN_WINDOW:i]
    test_df = df.iloc[i:i + STEP_SIZE]

    X_train, y_train = train_df[features], train_df['target']
    X_test = test_df[features]

    # Robust "Stump" Model
    last_model = xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.02, random_state=42)
    last_model.fit(X_train, y_train)

    probs = last_model.predict_proba(X_test)[:, 1]

    batch_results = pd.DataFrame({
        'date': test_df.index,
        'actual_ret': test_df['next_ret'],
        'prob': probs,
        'target': test_df['target']
    }, index=test_df.index)
    results_list.append(batch_results)

results = pd.concat(results_list)

# --- BACKTEST WITH TRANSACTION COSTS ---
# Scaling: 0.50 prob = 0%, 0.55 prob = 100%
results['position'] = np.clip((results['prob'] - 0.50) / 0.05, 0, 1)

# Costs & Returns
results['turnover'] = results['position'].diff().abs().fillna(0)
results['cost'] = results['turnover'] * TRANSACTION_COST
results['strat_ret'] = (results['position'] * results['actual_ret']) - results['cost']

results['cum_market'] = (1 + results['actual_ret']).cumprod()
results['cum_strat'] = (1 + results['strat_ret']).cumprod()

# --- NEW: PERFORMANCE METRICS ---
sharpe = (results['strat_ret'].mean() / results['strat_ret'].std()) * np.sqrt(252)
market_sharpe = (results['actual_ret'].mean() / results['actual_ret'].std()) * np.sqrt(252)

print("\n" + "=" * 30)
print(f"BACKTEST RESULTS (WFO)")
print("=" * 30)
print(f"WFO Accuracy:      {accuracy_score(results['target'], (results['prob'] > 0.5)):.2%}")
print(f"Strategy Sharpe:   {sharpe:.2f}")
print(f"Market Sharpe:     {market_sharpe:.2f}")
print(f"Strategy Return:   {(results['cum_strat'].iloc[-1] - 1):.2%}")
print(f"Market Return:     {(results['cum_market'].iloc[-1] - 1):.2%}")
print("=" * 30)


# --- LIVE PREDICTION FUNCTION ---
def get_live_signal(current_model):
    print("\n--- LIVE SIGNAL GENERATOR ---")
    raw_live = yf.download([SYMBOL] + MACRO_TICKERS, period="1y", auto_adjust=False)
    live_closes = raw_live['Close'].ffill()
    live_closes.columns = ['DXY', 'GSPC', 'TNX', 'VIX']
    live_open = raw_live['Open'][SYMBOL].ffill()

    row = {
        'rsi': (live_closes['GSPC'] > live_closes['GSPC'].shift(1)).rolling(14).mean().iloc[-1],
        'overnight_gap': (live_open.iloc[-1] - live_closes['GSPC'].iloc[-2]) / live_closes['GSPC'].iloc[-2],
        'dist_sma200': (live_closes['GSPC'].iloc[-1] - live_closes['GSPC'].rolling(200).mean().iloc[-1]) /
                       live_closes['GSPC'].rolling(200).mean().iloc[-1],
        'vix_ret': live_closes['VIX'].pct_change().iloc[-1],
        'tnx_ret': live_closes['TNX'].pct_change().iloc[-1],
        'ret_lag_1': live_closes['GSPC'].pct_change().iloc[-1]
    }

    X_live = pd.DataFrame([row])[features]
    prob_up = current_model.predict_proba(X_live)[0, 1]

    print(f"Latest Market Date: {live_closes.index[-1].date()}")
    print(f"Probability S&P 500 closes HIGHER tomorrow: {prob_up:.2%}")
    if prob_up > 0.52:
        print("SIGNAL: BUY / GO LONG")
    elif prob_up < 0.48:
        print("SIGNAL: SELL / STAY IN CASH")
    else:
        print("SIGNAL: NEUTRAL / WAIT")


# --- PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(results['date'], results['cum_market'], label=f'S&P 500 (Sharpe: {market_sharpe:.2f})', color='gray',
         alpha=0.5)
plt.plot(results['date'], results['cum_strat'], label=f'WFO Strategy (Sharpe: {sharpe:.2f})', color='blue')
plt.title("Industrial Backtest: Walk-Forward + Transaction Costs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

if last_model:
    get_live_signal(last_model)