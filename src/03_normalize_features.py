from pathlib import Path
import json

import pandas as pd


PROCESSED_DIR = Path("data/processed")

FEATURES_FILE = PROCESSED_DIR / "features_weekly.csv"
RETURNS_FILE = PROCESSED_DIR / "returns_weekly.csv"

FEATURES_NORM_FILE = PROCESSED_DIR / "features_weekly_normalized.csv"
RETURNS_ALIGNED_FILE = PROCESSED_DIR / "returns_weekly_aligned.csv"
SCALER_FILE = PROCESSED_DIR / "train_scaler_params.json"


TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"


def main():
    features = pd.read_csv(FEATURES_FILE, index_col="date", parse_dates=True)
    returns = pd.read_csv(RETURNS_FILE, index_col="date", parse_dates=True)

    # Alinear features y retornos por fecha
    common_index = features.index.intersection(returns.index)
    features = features.loc[common_index].copy()
    returns = returns.loc[common_index].copy()

    # Splits temporales
    train_mask = features.index <= TRAIN_END
    val_mask = (features.index > TRAIN_END) & (features.index <= VAL_END)
    test_mask = features.index > VAL_END

    train_features = features.loc[train_mask]

    if train_features.empty:
        raise ValueError("El bloque de entrenamiento está vacío. Revisa TRAIN_END o las fechas de los datos.")

    # Media y desviación estándar SOLO usando train
    train_mean = train_features.mean()
    train_std = train_features.std()

    # Evitar división por cero
    train_std = train_std.replace(0, 1.0)

    # Normalizar toda la muestra usando parámetros del train
    features_norm = (features - train_mean) / train_std

    # Guardar archivos
    features_norm.to_csv(FEATURES_NORM_FILE)
    returns.to_csv(RETURNS_ALIGNED_FILE)

    scaler_params = {
        "train_start": str(features.loc[train_mask].index.min().date()),
        "train_end": str(features.loc[train_mask].index.max().date()),
        "validation_start": str(features.loc[val_mask].index.min().date()) if val_mask.any() else None,
        "validation_end": str(features.loc[val_mask].index.max().date()) if val_mask.any() else None,
        "test_start": str(features.loc[test_mask].index.min().date()) if test_mask.any() else None,
        "test_end": str(features.loc[test_mask].index.max().date()) if test_mask.any() else None,
        "mean": train_mean.to_dict(),
        "std": train_std.to_dict(),
    }

    with open(SCALER_FILE, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=4)

    print("Normalización completada.")
    print(f"Features normalizadas: {FEATURES_NORM_FILE}")
    print(f"Returns alineados: {RETURNS_ALIGNED_FILE}")
    print(f"Parámetros scaler: {SCALER_FILE}")

    print("\nShapes:")
    print(f"Features: {features_norm.shape}")
    print(f"Returns:  {returns.shape}")

    print("\nFechas:")
    print(
        "Train:",
        features.loc[train_mask].index.min().date(),
        "->",
        features.loc[train_mask].index.max().date(),
    )

    if val_mask.any():
        print(
            "Validation:",
            features.loc[val_mask].index.min().date(),
            "->",
            features.loc[val_mask].index.max().date(),
        )
    else:
        print("Validation: vacío")

    if test_mask.any():
        print(
            "Test:",
            features.loc[test_mask].index.min().date(),
            "->",
            features.loc[test_mask].index.max().date(),
        )
    else:
        print("Test: vacío")

    print("\nControl de normalización en train:")
    print("Media train normalizada, primeras 5 variables:")
    print(features_norm.loc[train_mask].mean().round(6).head())

    print("\nStd train normalizada, primeras 5 variables:")
    print(features_norm.loc[train_mask].std().round(6).head())

    print("\nNaN totales en features normalizadas:")
    print(int(features_norm.isna().sum().sum()))

    print("\nNaN totales en returns:")
    print(int(returns.isna().sum().sum()))


if __name__ == "__main__":
    main()python src/03_normalize_features.py