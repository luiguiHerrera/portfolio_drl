from pathlib import Path
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "prices_all_assets.csv"

FEATURES_FILE = PROCESSED_DIR / "features_weekly.csv"
RETURNS_FILE = PROCESSED_DIR / "returns_weekly.csv"


def compute_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    return drawdown


def load_daily_close_prices(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f"No existe {input_file}. Ejecuta primero: python src/01_download_prices.py"
        )

    prices = pd.read_csv(input_file, parse_dates=["date"])
    required_cols = {"date", "ticker", "adj_close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {input_file}: {missing}")

    prices_daily = prices.pivot_table(
        index="date",
        columns="ticker",
        values="adj_close",
        aggfunc="last",
    )
    return prices_daily.sort_index()


def main():
    prices_daily = load_daily_close_prices(INPUT_FILE)

    # Precios semanales: cierre del viernes o último dato disponible de la semana
    prices_weekly = prices_daily.resample("W-FRI").last().ffill()

    log_returns = np.log(prices_weekly / prices_weekly.shift(1))

    features = pd.DataFrame(index=prices_weekly.index)

    for asset in prices_weekly.columns:
        r = log_returns[asset]

        features[f"{asset}_ret_1w"] = r.shift(1)
        features[f"{asset}_mom_4w"] = prices_weekly[asset] / prices_weekly[asset].shift(4) - 1.0
        features[f"{asset}_mom_12w"] = prices_weekly[asset] / prices_weekly[asset].shift(12) - 1.0
        features[f"{asset}_vol_4w"] = r.rolling(4).std()
        features[f"{asset}_vol_12w"] = r.rolling(12).std()

    dd = compute_drawdown(prices_weekly)

    for asset in prices_weekly.columns:
        features[f"{asset}_dd"] = dd[asset]

    # Target operacional para el entorno: retorno realizado siguiente periodo.
    future_returns = log_returns.shift(-1)

    features = features.dropna()
    future_returns = future_returns.loc[features.index].dropna()

    # Alinear de nuevo por si se pierde la última fila
    common_index = features.index.intersection(future_returns.index)
    features = features.loc[common_index]
    future_returns = future_returns.loc[common_index]

    features.to_csv(FEATURES_FILE)
    future_returns.to_csv(RETURNS_FILE)

    print("Features guardadas en:", FEATURES_FILE)
    print("Retornos futuros guardados en:", RETURNS_FILE)
    print("Features shape:", features.shape)
    print("Returns shape:", future_returns.shape)
    print("\nÚltimas filas features:")
    print(features.tail())


if __name__ == "__main__":
    main()
