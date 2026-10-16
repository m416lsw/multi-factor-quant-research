# multi-factor-quant-research
A Python-based multi-factor equity research and backtesting framework combining momentum, mean reversion, low volatility, and trend signals.

# Multi-Factor Quantitative Research System

A Python-based quantitative equity research framework that combines factor investing, portfolio optimization, historical backtesting, and Monte Carlo risk simulation.

## Overview

This project investigates whether a systematic multi-factor strategy can construct an equity portfolio using four signals:

- **Momentum:** Preference for stocks with strong medium-term returns.
- **Mean Reversion:** Preference for stocks with weaker short-term returns.
- **Low Volatility:** Preference for stocks exhibiting lower historical return volatility.
- **Trend:** Preference for stocks trading above their long-term moving averages.

The objective is to explore factor-based stock selection, portfolio construction, and risk-adjusted performance through a reproducible Python research workflow.

## Research Methodology

### 1. Data Collection
- Historical adjusted daily prices obtained using `yfinance`.
- Configurable equity universe and backtesting period.
- Daily returns calculated from adjusted closing prices.

### 2. Factor Construction

Four quantitative factors are calculated:

| Factor | Definition | Lookback |
|---|---|---|
| Momentum | Price return | 126 trading days |
| Mean Reversion | Negative short-term return | 5 trading days |
| Low Volatility | Negative annualized return volatility | 63 trading days |
| Trend | Price relative to moving average | 200 trading days |

Each factor is standardized using cross-sectional z-scores.

The composite score is calculated as:

\[
S_{i,t}=\sum_{k=1}^{4}w_k Z_{i,k,t}
\]

where each factor initially receives an equal weight of 25%.

### 3. Portfolio Construction
- Select the five highest-scoring stocks.
- Estimate the covariance matrix using historical daily returns.
- Optimize portfolio weights by minimizing portfolio variance.
- Apply long-only constraints.
- Cap individual position weights at 30%.

### 4. Backtesting
- Historical portfolio simulation.
- Rebalancing every 21 trading days.
- Signals generated using information available through the previous trading day's close.
- Transaction costs modeled at 10 basis points per unit of turnover.

### 5. Performance Evaluation

The framework calculates:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Historical Value at Risk (VaR)
- Expected Shortfall (ES)

### 6. Monte Carlo Simulation

Historical daily portfolio returns are resampled using bootstrap simulation to explore possible future portfolio outcomes.

Default parameters:
- 1,000 simulations
- 252-trading-day horizon
- Initial capital of $1,000,000

## Technology Stack

- Python
- NumPy
- Pandas
- SciPy
- yfinance
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/multi-factor-quant-research.git
cd multi-factor-quant-research
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the research pipeline:

```bash
python main.py
```

## Research Limitations

This project is an educational research prototype rather than a production-ready trading system.

Current limitations include:

- A manually selected stock universe that may introduce survivorship and selection bias.
- Reliance on historical adjusted closing prices.
- Simplified transaction-cost assumptions.
- No explicit market-impact or liquidity model.
- A constant risk-free rate assumption.
- Independent bootstrap sampling that does not preserve serial dependence or volatility clustering.
- Portfolio execution timing and missing-data handling requiring further validation.

Consequently, historical backtest results should not be interpreted as evidence of live trading profitability.

## Future Improvements

- [ ] Implement point-in-time historical index constituents.
- [ ] Correct and validate trade execution timing.
- [ ] Add benchmark comparisons against SPY.
- [ ] Introduce walk-forward and out-of-sample testing.
- [ ] Incorporate rolling risk-free rates.
- [ ] Evaluate factor performance using Information Coefficients.
- [ ] Implement factor attribution analysis.
- [ ] Add transaction-cost sensitivity analysis.
- [ ] Introduce block-bootstrap Monte Carlo simulation.
- [ ] Develop automated unit tests.

## Disclaimer

This project is intended for educational and quantitative research purposes only. It does not constitute investment advice.
