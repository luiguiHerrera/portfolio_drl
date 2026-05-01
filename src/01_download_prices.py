from pathlib import Path
import pandas as pd
import yfinance as yf


TICKERS = ["SPY", "GLD", "TLT", "BTC-USD"]

START_DATE = "2015-01-01"
END_DATE = None  # None = hasta la fecha disponible más reciente


def download_prices(ticker: str) -> pd.DataFrame:
    print(f"Descargando {ticker}...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No se descargaron datos para {ticker}")

    # Si yfinance devuelve columnas MultiIndex, las aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Normalizar nombres
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

    # Validación mínima
    required_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas en {ticker}: {missing}")

    df["ticker"] = ticker

    return df


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for ticker in TICKERS:
        df = download_prices(ticker)

        file_name = ticker.lower().replace("-", "_")
        raw_path = raw_dir / f"{file_name}.csv"

        df.to_csv(raw_path, index=False)
        print(f"Guardado: {raw_path}")

        all_data.append(df)

    prices = pd.concat(all_data, ignore_index=True)

    output_path = processed_dir / "prices_all_assets.csv"
    prices.to_csv(output_path, index=False)

    print("\nDescarga completada.")
    print(f"Archivo consolidado: {output_path}")
    print(prices.head())


if __name__ == "__main__":
    main()
