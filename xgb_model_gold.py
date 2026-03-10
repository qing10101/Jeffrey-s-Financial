import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
SYMBOL = "GC=F"
# Add NASDAQ (^IXIC) as a Liquidity Proxy
MACRO_TICKERS = ["TIP", "DX-Y.NYB", "SI=F", "^VIX", "^IXIC"]
TRANSACTION_COST = 0.0001
TRAIN_WINDOW = 1200  # Longer window for commodity cycles
STEP_SIZE = 30


def get_gold_pro_data():
    print(f"Fetching Gold and Liquidity data...")
    raw = yf.download([SYMBOL] + MACRO_TICKERS, period="max", auto_adjust=False)
    closes = raw['Close'].ffill()

    df = pd.DataFrame(index=closes.index)
    df['gold'] = closes[SYMBOL]
    df['tip'] = closes['TIP']
    df['dxy'] = closes['DX-Y.NYB']
    df['silver'] = closes['SI=F']
    df['nasdaq'] = closes['^IXIC']

    # --- PRO FEATURE ENGINEERING ---
    df['ret'] = df['gold'].pct_change()
    df['tip_ret'] = df['tip'].pct_change()
    df['dxy_ret'] = df['dxy'].pct_change()
    df['nasdaq_ret'] = df['nasdaq'].pct_change()

    # 1. Macro Divergence: Is Gold outperforming the Dollar?
    df['gold_vs_dxy'] = df['ret'] - df['dxy_ret']

    # 2. Volatility (ATR Proxy)
    df['volatility'] = df['ret'].rolling(20).std()

    # 3. Gold/Silver Ratio Momentum
    df['gs_mom'] = (df['gold'] / df['silver']).pct_change(5)

    # 4. Long-Term Trend Filter
    df['sma_100'] = df['gold'].rolling(100).mean()
    df['is_uptrend'] = (df['gold'] > df['sma_100']).astype(int)

    # 5. Lags
    df['ret_lag_1'] = df['ret'].shift(1)
    df['tip_lag_1'] = df['tip_ret'].shift(1)

    # Targets
    df['target'] = (df['gold'].shift(-1) > df['gold']).astype(int)
    df['next_ret'] = df['ret'].shift(-1)

    return df.dropna()


df = get_gold_pro_data()
features = ['tip_ret', 'dxy_ret', 'nasdaq_ret', 'gold_vs_dxy', 'volatility', 'gs_mom', 'is_uptrend', 'ret_lag_1']

# --- WALK-FORWARD LOOP ---
results_list = []
start_idx = TRAIN_WINDOW
last_model = None

print(f"Executing Pro Gold Backtest...")
for i in range(start_idx, len(df) - STEP_SIZE, STEP_SIZE):
    train_df = df.iloc[i - TRAIN_WINDOW:i]
    test_df = df.iloc[i:i + STEP_SIZE]

    last_model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.01, subsample=0.8, random_state=42)
    last_model.fit(train_df[features], train_df['target'])

    probs = last_model.predict_proba(test_df[features])[:, 1]

    batch = pd.DataFrame({
        'date': test_df.index,
        'actual_ret': test_df['next_ret'],
        'prob': probs,
        'is_uptrend': test_df['is_uptrend'],
        'target': test_df['target']
    }, index=test_df.index)
    results_list.append(batch)

results = pd.concat(results_list)

# --- THE "TREND-FOLLOWING" POSITION LOGIC ---
# If the trend is UP, we only need 49% probability to stay in.
# This prevents the model from "over-trading" during a bull run.
results['threshold'] = np.where(results['is_uptrend'] == 1, 0.49, 0.53)
results['position'] = (results['prob'] > results['threshold']).astype(float)

# Smoothing the position to reduce turnover costs
results['position'] = results['position'].rolling(3).mean().fillna(0)

results['cost'] = results['position'].diff().abs().fillna(0) * TRANSACTION_COST
results['strat_ret'] = (results['position'] * results['actual_ret']) - results['cost']

results['cum_market'] = (1 + results['actual_ret']).cumprod()
results['cum_strat'] = (1 + results['strat_ret']).cumprod()

# --- EVALUATION ---
sharpe = (results['strat_ret'].mean() / results['strat_ret'].std()) * np.sqrt(252)
print("\n" + "=" * 30)
print(f"PRO GOLD RESULTS")
print("=" * 30)
print(f"Accuracy:      {accuracy_score(results['target'], (results['prob'] > 0.5)):.2%}")
print(f"Sharpe Ratio:  {sharpe:.2f}")
print(f"Strategy Ret:  {(results['cum_strat'].iloc[-1] - 1):.2%}")
print(f"Market Ret:    {(results['cum_market'].iloc[-1] - 1):.2%}")

plt.figure(figsize=(12, 6))
plt.plot(results['date'], results['cum_market'], label='Gold Spot', color='gold', alpha=0.4)
plt.plot(results['date'], results['cum_strat'], label='Pro ML Gold Strategy', color='black', linewidth=1.5)
plt.title("Gold Strategy: Trend-Following + Liquidity Proxy")
plt.legend()
plt.show()

# --- LIVE SIGNAL ---
if last_model:
    raw_l = yf.download([SYMBOL] + MACRO_TICKERS, period="1y", auto_adjust=False)
    l_c = raw_l['Close'].ffill()

    row = {
        'tip_ret': l_c['TIP'].pct_change().iloc[-1],
        'dxy_ret': l_c['DX-Y.NYB'].pct_change().iloc[-1],
        'nasdaq_ret': l_c['^IXIC'].pct_change().iloc[-1],
        'gold_vs_dxy': l_c[SYMBOL].pct_change().iloc[-1] - l_c['DX-Y.NYB'].pct_change().iloc[-1],
        'volatility': l_c[SYMBOL].pct_change().rolling(20).std().iloc[-1],
        'gs_mom': (l_c[SYMBOL] / l_c['SI=F']).pct_change(5).iloc[-1],
        'is_uptrend': int(l_c[SYMBOL].iloc[-1] > l_c[SYMBOL].rolling(100).mean().iloc[-1]),
        'ret_lag_1': l_c[SYMBOL].pct_change().iloc[-1]
    }

    prob_live = last_model.predict_proba(pd.DataFrame([row])[features])[0, 1]
    print(f"\nLive Signal Date: {l_c.index[-1].date()}")
    print(f"Next Day Up-Prob: {prob_live:.2%}")