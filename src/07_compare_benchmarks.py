from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RETURNS_FILE = PROCESSED_DIR / "returns_weekly_aligned.csv"
POLICY_WEIGHTS_FILE = TABLES_DIR / "policy_weights_all.csv"

SUMMARY_FILE = TABLES_DIR / "benchmark_summary.csv"
EQUITY_CURVES_FILE = TABLES_DIR / "equity_curves.csv"
FIGURE_FILE = FIGURES_DIR / "equity_curves.png"


TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (peak - equity_curve) / peak
    return float(dd.max())


def annualized_return(weekly_returns: pd.Series) -> float:
    n_weeks = len(weekly_returns)
    if n_weeks == 0:
        return np.nan

    total_return = float(np.exp(weekly_returns.sum()) - 1.0)
    years = n_weeks / 52

    if years <= 0:
        return np.nan

    return float((1 + total_return) ** (1 / years) - 1)


def annualized_volatility(weekly_returns: pd.Series) -> float:
    return float(weekly_returns.std() * np.sqrt(52))


def sharpe_ratio(weekly_returns: pd.Series, rf_annual: float = 0.0) -> float:
    rf_weekly = np.log(1 + rf_annual) / 52
    excess = weekly_returns - rf_weekly
    vol = excess.std()

    if vol == 0 or np.isnan(vol):
        return np.nan

    return float((excess.mean() / vol) * np.sqrt(52))


def performance_metrics(weekly_returns: pd.Series, name: str) -> dict:
    equity = np.exp(weekly_returns.cumsum())

    total_return = float(equity.iloc[-1] - 1.0)

    return {
        "strategy": name,
        "weeks": len(weekly_returns),
        "total_return": total_return,
        "annualized_return": annualized_return(weekly_returns),
        "annualized_volatility": annualized_volatility(weekly_returns),
        "sharpe": sharpe_ratio(weekly_returns),
        "max_drawdown": max_drawdown(equity),
    }


def build_benchmark_returns(returns: pd.DataFrame) -> pd.DataFrame:
    assets = list(returns.columns)

    benchmark_returns = pd.DataFrame(index=returns.index)

    # 1. Equal Weight: 25% cada activo
    equal_weights = np.ones(len(assets)) / len(assets)
    benchmark_returns["Equal_Weight"] = returns.values @ equal_weights

    # 2. 60/40 tradicional: 60% SPY, 40% TLT
    if "SPY" in assets and "TLT" in assets:
        w_6040 = np.zeros(len(assets))
        w_6040[assets.index("SPY")] = 0.60
        w_6040[assets.index("TLT")] = 0.40
        benchmark_returns["SPY_TLT_60_40"] = returns.values @ w_6040

    # 3. Buy & Hold individual por activo
    for asset in assets:
        benchmark_returns[f"BuyHold_{asset}"] = returns[asset]

    return benchmark_returns


def load_drl_returns():
    weights_df = pd.read_csv(POLICY_WEIGHTS_FILE, index_col="date", parse_dates=True)

    if "portfolio_return" not in weights_df.columns:
        raise ValueError("No se encontró la columna portfolio_return en policy_weights_all.csv")

    return weights_df["portfolio_return"].rename("DRL_Policy")


def evaluate_period(name: str, returns_df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    data = returns_df.copy()

    if start_date is not None:
        data = data.loc[data.index >= pd.to_datetime(start_date)]

    if end_date is not None:
        data = data.loc[data.index <= pd.to_datetime(end_date)]

    rows = []

    for col in data.columns:
        metrics = performance_metrics(data[col].dropna(), col)
        metrics["period"] = name
        rows.append(metrics)

    return pd.DataFrame(rows)


def main():
    returns = pd.read_csv(RETURNS_FILE, index_col="date", parse_dates=True)

    drl_returns = load_drl_returns()

    benchmark_returns = build_benchmark_returns(returns)

    all_returns = benchmark_returns.join(drl_returns, how="inner")

    # Guardar curvas de capital
    equity_curves = np.exp(all_returns.cumsum())
    equity_curves.to_csv(EQUITY_CURVES_FILE)

    # Métricas por periodo
    summary_parts = [
        evaluate_period("Train", all_returns, end_date=TRAIN_END),
        evaluate_period("Validation", all_returns, start_date="2022-01-01", end_date=VAL_END),
        evaluate_period("Test", all_returns, start_date="2024-01-01"),
        evaluate_period("Full", all_returns),
    ]

    summary = pd.concat(summary_parts, ignore_index=True)

    # Orden útil
    summary = summary[
        [
            "period",
            "strategy",
            "weeks",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
        ]
    ]

    summary.to_csv(SUMMARY_FILE, index=False)

    print("Resumen de benchmarks guardado en:", SUMMARY_FILE)
    print("Curvas de capital guardadas en:", EQUITY_CURVES_FILE)

    print("\nResumen TEST ordenado por Sharpe:")
    test_summary = summary[summary["period"] == "Test"].sort_values("sharpe", ascending=False)
    print(test_summary.round(4).to_string(index=False))
    
    # Gráfico de curvas en test, rebased a 1
    test_returns = all_returns.loc[all_returns.index >= "2024-01-01"]
    test_equity = np.exp(test_returns.cumsum())
    plt.figure(figsize=(12, 7))


    for col in test_equity.columns:
        plt.plot(test_equity.index, test_equity[col], label=col)

    plt.title("Equity curves - Test period")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)

    print("\nGráfico guardado en:", FIGURE_FILE)


if __name__ == "__main__":
    main()