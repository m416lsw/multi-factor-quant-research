"""
Multi-Factor Research System

Strategy: Momentum + Mean Reversion + Low Volatility + Trend

Pipeline:
    1. Market data (yfinance)
    2. Factor construction (cross-sectional z-score)
    3. Top-N selection + min-variance weights (cap per position)
    4. Transaction cost modeling
    5. Backtest (signal at t-1 close, trade at t)
    6. Performance / risk analytics
    7. Monte Carlo (bootstrap)
    8. Visualization
"""
#"매달 말일에 "모멘텀·저변동·추세 점수 상위 5개를 산다"는 규칙을, 그 시점에 알 수 있던 정보만으로 따라갔다면 어땠는지"
#"이왕이면 “2015년 당시 S&P 500 구성종목”처럼 그 시점 기준으로 짜는게 좋음"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from dataclasses import dataclass
from scipy.optimize import minimize


# ============================================================
# 1. CONFIGURATION
# config에 종목티커 넣기. minimum 50개.
# ============================================================

@dataclass
class Config:
    tickers = (
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AVGO", "AMD", "QCOM",
        "INTC", "TXN", "MU", "AMAT", "LRCX",
        "KLAC", "TSM", "ASML", "MRVL", "ON",
        "ADI", "ORCL", "CRM", "ADBE", "NOW",
        "INTU", "PANW", "CRWD", "SNOW", "DDOG",
        "NET", "PLTR", "WDAY", "FTNT", "ZS",
        "CSCO", "IBM", "HPE", "DELL", "ANET",
        "SMCI", "NFLX", "UBER", "SHOP", "PYPL",
        "ABNB", "COIN",
    )

    start = "2021-01-01"
    end = "2026-10-16"

    initial_capital = 1_000_000

    # 영업일 기준 window
    momentum_window = 126        # ~6개월
    reversal_window = 5          # ~1주
    volatility_window = 63       # ~1분기
    moving_average_window = 200  # 장기 이평선

    # 팩터 가중치: 근거 없는 값 튜닝하면 과최적화 → equal weight로 시작
    factor_weights = {
        "momentum": 0.25,
        "reversal": 0.25,
        "volatility": 0.25,
        "trend": 0.25,
    }

    top_n = 5
    rebalance_frequency = 21     # ~1개월

    transaction_cost_bps = 10

    annual_risk_free_rate = 0.03  # TODO: US 10Y yield 시계열로 교체

    max_position_weight = 0.30    # TODO: TAA 고려해서 가변화


CFG = Config()


# ============================================================
# 2. MARKET DATA ENGINE
# ============================================================

class MarketDataEngine:

    def __init__(self, config):
        self.config = config
        self.prices = None

    def download(self):

        raw = yf.download(
            list(self.config.tickers),
            start=self.config.start,
            end=self.config.end,
            auto_adjust=True,
            progress=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = [self.config.tickers[0]]

        prices = prices.sort_index().ffill()

        self.prices = prices
        return prices


# ============================================================
# 3. FACTOR RESEARCH ENGINE
# ============================================================

class FactorEngine:

    def __init__(self, prices):
        self.prices = prices
        self.returns = prices.pct_change(fill_method=None)

    @staticmethod
    def cross_sectional_zscore(frame):

        mean = frame.mean(axis=1)
        std = frame.std(axis=1)

        return frame.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

    def momentum_factor(self):

        momentum = self.prices / self.prices.shift(CFG.momentum_window) - 1
        return self.cross_sectional_zscore(momentum)

    def reversal_factor(self):

        short_term_return = self.prices / self.prices.shift(CFG.reversal_window) - 1
        return self.cross_sectional_zscore(-short_term_return)

    def volatility_factor(self):

        # sqrt(252): 일간 → 연율화. 순위엔 영향 없지만 관행상 유지
        volatility = self.returns.rolling(CFG.volatility_window).std() * np.sqrt(252)
        return self.cross_sectional_zscore(-volatility)  # 저변동성 선호

    def trend_factor(self):

        moving_average = self.prices.rolling(CFG.moving_average_window).mean()
        trend = self.prices / moving_average - 1
        return self.cross_sectional_zscore(trend)

    def composite_factor(self):

        w = CFG.factor_weights

        factors = {
            "momentum": self.momentum_factor(),
            "reversal": self.reversal_factor(),
            "volatility": self.volatility_factor(),
            "trend": self.trend_factor(),
        }

        composite = sum(w[name] * frame for name, frame in factors.items())

        return composite


# ============================================================
# 4. PORTFOLIO OPTIMIZATION
# ============================================================

class PortfolioOptimizer:

    def __init__(self, max_weight=0.30):
        self.max_weight = max_weight

    def minimum_variance(self, covariance):

        n_assets = covariance.shape[0]
        initial_weights = np.ones(n_assets) / n_assets

        def objective(weights):
            return weights @ covariance @ weights

        bounds = [(0, self.max_weight)] * n_assets

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            return initial_weights

        return result.x


# ============================================================
# 5. BACKTESTING ENGINE
# ============================================================

class BacktestEngine:

    def __init__(self, prices, signals, config):

        self.prices = prices
        self.signals = signals
        self.config = config
        self.returns = prices.pct_change(fill_method=None)

        self.optimizer = PortfolioOptimizer(config.max_position_weight)

    def generate_target_weights(self, date):

        scores = self.signals.loc[date].dropna()

        if len(scores) < self.config.top_n:
            return pd.Series(0.0, index=self.prices.columns)

        selected = scores.nlargest(self.config.top_n).index

        historical_returns = (
            self.returns.loc[:date, selected].tail(126).dropna()
        )

        if len(historical_returns) < 60:
            optimized_weights = np.ones(len(selected)) / len(selected)
        else:
            covariance = historical_returns.cov().values
            covariance = covariance + np.eye(len(selected)) * 1e-8
            optimized_weights = self.optimizer.minimum_variance(covariance)

        target = pd.Series(0.0, index=self.prices.columns)
        target.loc[selected] = optimized_weights

        return target

    def run(self):

        dates = self.prices.index

        portfolio_returns = pd.Series(0.0, index=dates)
        weights_history = pd.DataFrame(0.0, index=dates, columns=self.prices.columns)
        turnover_history = pd.Series(0.0, index=dates)
        current_weights = pd.Series(0.0, index=self.prices.columns)

        # factor가 NaN 아닌 첫 시점 (MA 200일)
        start_index = self.config.moving_average_window

        for i in range(start_index, len(dates)):

            date = dates[i]
            previous_date = dates[i - 1]

            daily_asset_returns = self.returns.loc[date].fillna(0.0)

            gross_return = float(current_weights @ daily_asset_returns)

            # 시장 움직임 후 weight drift
            position_values = current_weights * (1 + daily_asset_returns)
            denominator = 1 + gross_return

            if denominator > 0:
                drifted_weights = position_values / denominator
            else:
                drifted_weights = current_weights.copy()

            transaction_cost = 0.0  # 매일 리셋 (rebalance 날에만 발생)

            if (i - start_index) % self.config.rebalance_frequency == 0:

                # 신호는 전일 종가 기준
                target_weights = self.generate_target_weights(previous_date)

                turnover = float(np.abs(target_weights - drifted_weights).sum())

                transaction_cost = (
                    turnover * self.config.transaction_cost_bps / 10000
                )

                turnover_history.loc[date] = turnover
                current_weights = target_weights

            else:
                current_weights = drifted_weights

            portfolio_returns.loc[date] = gross_return - transaction_cost
            weights_history.loc[date] = current_weights

        equity_curve = self.config.initial_capital * (1 + portfolio_returns).cumprod()

        return {
            "returns": portfolio_returns.iloc[start_index:],
            "equity": equity_curve.iloc[start_index:],
            "weights": weights_history.iloc[start_index:],
            "turnover": turnover_history.iloc[start_index:],
        }


# ============================================================
# 6. PERFORMANCE ANALYTICS
# ============================================================

class PerformanceAnalyzer:

    def __init__(self, returns, risk_free_rate=0.03):

        self.returns = returns.dropna()
        self.rf = risk_free_rate

    def annualized_return(self):

        growth = (1 + self.returns).prod()
        years = len(self.returns) / 252

        return growth ** (1 / years) - 1

    def annualized_volatility(self):

        return self.returns.std() * np.sqrt(252)

    def sharpe_ratio(self):

        volatility = self.annualized_volatility()

        if volatility == 0:
            return np.nan

        return (self.annualized_return() - self.rf) / volatility

    def sortino_ratio(self):

        daily_rf = (1 + self.rf) ** (1 / 252) - 1

        downside = np.minimum(self.returns - daily_rf, 0)
        downside_deviation = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)

        if downside_deviation == 0:
            return np.nan

        return (self.annualized_return() - self.rf) / downside_deviation

    def maximum_drawdown(self):

        wealth = (1 + self.returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1

        return drawdown.min()

    def historical_var(self, confidence=0.95):

        return -np.quantile(self.returns, 1 - confidence)

    def expected_shortfall(self, confidence=0.95):

        threshold = np.quantile(self.returns, 1 - confidence)
        tail_returns = self.returns[self.returns <= threshold]

        return -tail_returns.mean()

    def calculate_all(self):

        return {
            "Annual Return": self.annualized_return(),
            "Annual Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Maximum Drawdown": self.maximum_drawdown(),
            "Daily VaR 95%": self.historical_var(),
            "Daily ES 95%": self.expected_shortfall(),
        }


# ============================================================
# 7. MONTE CARLO RISK SIMULATION
# ============================================================

class MonteCarloEngine:

    def __init__(self, returns, initial_capital):

        self.returns = returns.dropna()
        self.initial_capital = initial_capital

    def simulate(self, n_simulations=1000, horizon=252):

        simulated_paths = np.zeros((horizon + 1, n_simulations))
        simulated_paths[0] = self.initial_capital

        historical_returns = self.returns.values

        for simulation in range(n_simulations):

            sampled_returns = np.random.choice(
                historical_returns, size=horizon, replace=True
            )

            simulated_paths[1:, simulation] = (
                self.initial_capital * np.cumprod(1 + sampled_returns)
            )

        return simulated_paths


# ============================================================
# 8. VISUALIZATION ENGINE
# ============================================================

class VisualizationEngine:

    @staticmethod
    def plot_equity_curve(equity):

        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity.values, label="Multi-Factor Strategy")
        plt.title("Portfolio Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_drawdown(returns):

        wealth = (1 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1

        plt.figure(figsize=(12, 5))
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4)
        plt.title("Portfolio Drawdown")
        plt.ylabel("Drawdown")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_monte_carlo(paths):

        plt.figure(figsize=(12, 6))
        plt.plot(paths[:, :100], alpha=0.15)
        plt.title("Monte Carlo Portfolio Simulations")
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================
# 9. MAIN RESEARCH PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("QUANTITATIVE MULTI-FACTOR RESEARCH SYSTEM")
    print("=" * 65)

    print("\n[1] Downloading market data...")
    prices = MarketDataEngine(CFG).download()
    print("Data shape:", prices.shape)

    print("\n[2] Constructing alpha factors...")
    signals = FactorEngine(prices).composite_factor()
    print("Factor construction complete.")

    print("\n[3] Running portfolio backtest...")
    results = BacktestEngine(prices, signals, CFG).run()

    strategy_returns = results["returns"]
    equity_curve = results["equity"]
    print("Backtest complete.")

    print("\n[4] Calculating performance metrics...")
    metrics = PerformanceAnalyzer(
        strategy_returns, CFG.annual_risk_free_rate
    ).calculate_all()

    for metric, value in metrics.items():
        if "Ratio" in metric:
            print(f"{metric:25s}: {value:.3f}")
        else:
            print(f"{metric:25s}: {value:.2%}")

    print("\n[5] Running Monte Carlo simulations...")
    simulations = MonteCarloEngine(
        strategy_returns, CFG.initial_capital
    ).simulate(n_simulations=1000, horizon=252)

    final_values = simulations[-1]
    print("Median simulated terminal wealth:", f"${np.median(final_values):,.2f}")
    print("5th percentile terminal wealth:", f"${np.percentile(final_values, 5):,.2f}")

    print("\n[6] Generating research charts...")
    VisualizationEngine.plot_equity_curve(equity_curve)
    VisualizationEngine.plot_drawdown(strategy_returns)
    VisualizationEngine.plot_monte_carlo(simulations)

    print("\nResearch pipeline completed.")


if __name__ == "__main__":
    main()