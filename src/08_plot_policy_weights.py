from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]

TABLES_DIR = ROOT_DIR / "outputs" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_FILE = TABLES_DIR / "policy_weights_all.csv"
FIGURE_FILE = FIGURES_DIR / "policy_weights_test.png"

TEST_START = "2024-01-01"


def main():
    weights = pd.read_csv(WEIGHTS_FILE, index_col="date", parse_dates=True)

    weight_cols = [col for col in weights.columns if col.startswith("w_")]

    test_weights = weights.loc[weights.index >= TEST_START, weight_cols].copy()

    if test_weights.empty:
        raise ValueError("No hay datos de pesos para el periodo test.")

    plt.figure(figsize=(12, 7))

    for col in weight_cols:
        plt.plot(test_weights.index, test_weights[col], label=col.replace("w_", ""))

    plt.title("DRL Policy Weights - Test Period")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)

    print("Gráfico de pesos guardado en:", FIGURE_FILE)

    print("\nPesos promedio en test:")
    print(test_weights.mean().sort_values(ascending=False).round(4))

    print("\nÚltimos pesos:")
    print(test_weights.tail().round(4))


if __name__ == "__main__":
    main()