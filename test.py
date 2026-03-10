import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- EXPERIMENT CONFIGURATION ---
# Change these to test different ideas!
TICKER = "^GSPC"  # Try ^NDX, NVDA, or ETH-USD
TRAIN_WINDOW = 500  # How much history to learn from
STEP_SIZE = 20  # How often to re-train
HORIZON = 1  # Predict 1 day, 3 days, or 5 days?
MAX_DEPTH = 2  # Tree complexity
LEARNING_RATE = 0.02


def get_experimental_data(ticker):
    print(f"🔬 Fetching Lab Data for {ticker}...")
    # Fetch Asset + Standard Macro Suite
    raw = yf.download([ticker, "^VIX", "^TNX", "DX-Y.NYB"], period="max", auto_adjust=False)
    closes = raw['Close'].ffill()

    # Handle MultiIndex
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = closes.columns.get_level_values(1)

    df = pd.DataFrame(index=closes.index)
    df['price'] = closes[ticker]
    df['vix'] = closes['^VIX']
    df['tnx'] = closes['^TNX']
    df['dxy'] = closes['DX-Y.NYB']

    # --- EXPERIMENT ZONE: ADD NEW FEATURES HERE ---
    df['ret'] = df['price'].pct_change()

    # Feature 1: RSI
    df['rsi'] = (df['price'] > df['price'].shift(1)).rolling(14).mean()

    # Feature 2: Volatility Spike
    df['vol_spike'] = df['ret'].rolling(5).std() / df['ret'].rolling(20).std()

    # Feature 3: Dollar Momentum
    df['dxy_mom'] = df['dxy'].pct_change(5)

    # --- TARGET DEFINITION ---
    df['target'] = (df['price'].shift(-HORIZON) > df['price']).astype(int)
    df['next_ret'] = df['ret'].shift(-1)

    return df.dropna()


# 1. Prepare Data
df = get_experimental_data(TICKER)
features = ['rsi', 'vol_spike', 'dxy_mom', 'tnx']  # Select features to test

# 2. Walk-Forward Loop
results_list = []
print(f"🏃 Running Experiment: {len(df)} days...")

for i in range(TRAIN_WINDOW, len(df) - STEP_SIZE, STEP_SIZE):
    train_df = df.iloc[i - TRAIN_WINDOW:i]
    test_df = df.iloc[i:i + STEP_SIZE]

    # Initialize Model with Experimental Parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=42
    )

    model.fit(train_df[features], train_df['target'])
    probs = model.predict_proba(test_df[features])[:, 1]

    batch = pd.DataFrame({
        'date': test_df.index,
        'actual_ret': test_df['next_ret'],
        'prob': probs,
        'target': test_df['target']
    }, index=test_df.index)
    results_list.append(batch)

# 3. Analyze Results
res = pd.concat(results_list)
res['position'] = (res['prob'] > 0.52).astype(float)
res['strat_ret'] = res['position'] * res['actual_ret']

res['cum_mkt'] = (1 + res['actual_ret']).cumprod()
res['cum_strat'] = (1 + res['strat_ret']).cumprod()

# 4. Final Metrics
acc = accuracy_score(res['target'], (res['prob'] > 0.5))
sharpe = (res['strat_ret'].mean() / res['strat_ret'].std()) * np.sqrt(252)

print("\n" + "=" * 30)
print(f"LAB TEST RESULTS: {TICKER}")
print("=" * 30)
print(f"Accuracy:      {acc:.2%}")
print(f"Sharpe Ratio:  {sharpe:.2f}")
print(f"Total Return:  {(res['cum_strat'].iloc[-1] - 1):.2%}")
print(f"Market Return: {(res['cum_mkt'].iloc[-1] - 1):.2%}")
print("=" * 30)

# 5. Plot
plt.figure(figsize=(12, 6))
plt.plot(res['date'], res['cum_mkt'], label='Benchmark', color='gray', alpha=0.5)
plt.plot(res['date'], res['cum_strat'], label='Experimental Strategy', color='purple')
plt.title(f"Experimental Backtest: {TICKER} (Depth={MAX_DEPTH}, Horizon={HORIZON})")
plt.legend()
plt.show()

# 6. Feature Importance Check
# Shows you if your new features actually matter
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', title='Feature Importance (Last Window)')
plt.show()