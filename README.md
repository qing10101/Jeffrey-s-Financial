***

# Jeffrey's Financial: Quantitative Trading Suite

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![yfinance](https://img.shields.io/badge/Data-yfinance-gold.svg)](https://github.com/ranaroussi/yfinance)

A specialized machine learning toolkit for predicting daily price direction and managing risk-adjusted allocations for the **S&P 500**, **Bitcoin**, and **Gold**.

## 🚀 Project Overview

Predicting financial markets is notoriously difficult due to low signal-to-noise ratios. This project utilizes **XGBoost** classifiers and **Walk-Forward Optimization (WFO)** to move beyond simple "look-ahead" bias. Instead of chasing total returns, the suite focuses on **Sharpe Ratio optimization**—maximizing profit per unit of risk.

### Core Assets & Strategies
1.  **S&P 500 (`^GSPC`):** The "Industrial" model. Focuses on overnight gaps, VIX sentiment, and 10-year Treasury yields to identify equity regimes.
2.  **Bitcoin (`BTC-USD`):** The "Alpha-Max" model. Utilizes 3-day prediction horizons and dynamic leverage (up to 1.5x) to capture parabolic bull runs while de-risking during 80% drawdowns.
3.  **Gold (`GC=F`):** The "Recovery" model. A macro-heavy strategy focusing on Real Yields (TIP ETF) and the US Dollar (DXY) to identify flight-to-safety capital flows.

## 🛠️ Technical Methodology

*   **Walk-Forward Optimization:** Models are never "static." They re-train every 14-20 trading days using a rolling window, allowing the system to adapt to new market regimes.
*   **Feature Engineering:** Includes Macro-correlation (Bond yields, Dollar index), momentum acceleration, overnight gaps, and volatility Z-scores.
*   **Hysteresis Trading Logic:** To minimize transaction costs (slippage/fees), the models use "sticky" probability thresholds (e.g., 0.53 to buy, 0.48 to sell) to reduce unnecessary turnover.
*   **Transaction Cost Modeling:** All backtests include a mandatory 1-10 basis point penalty per trade to ensure results are realistic and tradable.

## 📈 Performance Summary

| Asset | Strategy | Accuracy | Sharpe Ratio | Market Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **S&P 500** | Industrial | 53.9% | 0.63 | Outperforms on Risk |
| **Bitcoin** | Alpha-Max | 52.1% | 0.84 | 1,135% vs 832% |
| **Gold** | Recovery | 52.6% | 0.70 | Captures Bull Cycles |

## 💻 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running a Backtest
To run the Bitcoin Alpha-Max strategy:
```bash
python xgb_model_btc.py
```

### Generating Live Signals
Each script includes a `get_live_signal` function. Run the script after the market close (4:15 PM EST) to generate a prediction for the following day:
```
# Output Example:
Probability S&P 500 closes HIGHER tomorrow: 61.15%
SIGNAL: BUY / GO LONG
```

## 📂 Repository Structure
*   `xgb_model_gspc.py`: High-Sharpe equity allocation model.
*   `xgb_model_btc.py`: Leveraged crypto-cycle model.
*   `xgb_model_gold.py`: Macro-based precious metals model.
*   `test.py`: Experimental script for feature testing.

## ⚠️ Disclaimer
**This project is for educational and research purposes only.** Financial markets involve significant risk. Past performance, especially in backtests, is not indicative of future results. Never trade money you cannot afford to lose.

---
*Developed as part of the Jeffrey's Financial research initiative into Quantitative Machine Learning.*
