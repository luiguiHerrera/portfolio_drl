from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.optim as optim


# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = PROCESSED_DIR / "features_weekly_normalized.csv"
RETURNS_FILE = PROCESSED_DIR / "returns_weekly_aligned.csv"

MODEL_FILE = MODELS_DIR / "policy_network_best.pt"
TRAIN_HISTORY_FILE = TABLES_DIR / "training_history.csv"
WEIGHTS_FILE = TABLES_DIR / "policy_weights_all.csv"


# =========================
# Import PolicyNetwork
# =========================

policy_path = SRC_DIR / "05_policy_network.py"

spec = importlib.util.spec_from_file_location("policy_network_module", policy_path)
policy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_module)

PolicyNetwork = policy_module.PolicyNetwork


# =========================
# Config
# =========================

TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"

EPOCHS = 500
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64

LAMBDA_VOL = 0.10
LAMBDA_DD = 0.20
LAMBDA_TO = 0.05

VOL_WINDOW = 4
INITIAL_VALUE = 1.0
SEED = 42


# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    features = pd.read_csv(FEATURES_FILE, index_col="date", parse_dates=True)
    returns = pd.read_csv(RETURNS_FILE, index_col="date", parse_dates=True)

    common_index = features.index.intersection(returns.index)
    features = features.loc[common_index].copy()
    returns = returns.loc[common_index].copy()

    return features, returns


def split_data(features: pd.DataFrame, returns: pd.DataFrame):
    train_mask = features.index <= TRAIN_END
    val_mask = (features.index > TRAIN_END) & (features.index <= VAL_END)
    test_mask = features.index > VAL_END

    splits = {
        "train": (features.loc[train_mask], returns.loc[train_mask]),
        "val": (features.loc[val_mask], returns.loc[val_mask]),
        "test": (features.loc[test_mask], returns.loc[test_mask]),
    }

    return splits


def to_tensor(df: pd.DataFrame):
    return torch.tensor(df.values, dtype=torch.float32)


def simulate_portfolio(
    model,
    features_tensor,
    returns_tensor,
    lambda_vol=LAMBDA_VOL,
    lambda_dd=LAMBDA_DD,
    lambda_to=LAMBDA_TO,
    vol_window=VOL_WINDOW,
    initial_value=INITIAL_VALUE,
):
    """
    Simulación diferenciable del portafolio.

    La red observa s_t, produce pesos w_t, recibe r_{t+1}
    y se calcula reward ajustado por retorno, volatilidad,
    drawdown y turnover.
    """

    n_periods = features_tensor.shape[0]
    n_assets = returns_tensor.shape[1]

    prev_weights = torch.ones(n_assets) / n_assets

    portfolio_value = torch.tensor(initial_value, dtype=torch.float32)
    peak_value = torch.tensor(initial_value, dtype=torch.float32)

    portfolio_returns = []
    rewards = []
    weights_history = []
    values_history = []
    drawdowns_history = []
    turnovers_history = []

    for t in range(n_periods):
        state_t = features_tensor[t]
        asset_returns_t = returns_tensor[t]

        logits = model(state_t)
        weights = torch.softmax(logits, dim=-1)

        portfolio_return = torch.sum(weights * asset_returns_t)

        portfolio_value = portfolio_value * torch.exp(portfolio_return)
        peak_value = torch.maximum(peak_value, portfolio_value)

        drawdown = (peak_value - portfolio_value) / peak_value
        turnover = torch.sum(torch.abs(weights - prev_weights))

        portfolio_returns.append(portfolio_return)

        if len(portfolio_returns) >= vol_window:
            recent_returns = torch.stack(portfolio_returns[-vol_window:])
            rolling_vol = torch.std(recent_returns, unbiased=False)
        else:
            rolling_vol = torch.tensor(0.0)

        reward = (
            portfolio_return
            - lambda_vol * rolling_vol
            - lambda_dd * drawdown
            - lambda_to * turnover
        )

        rewards.append(reward)
        weights_history.append(weights)
        values_history.append(portfolio_value)
        drawdowns_history.append(drawdown)
        turnovers_history.append(turnover)

        prev_weights = weights

    rewards = torch.stack(rewards)
    portfolio_returns = torch.stack(portfolio_returns)
    weights_history = torch.stack(weights_history)
    values_history = torch.stack(values_history)
    drawdowns_history = torch.stack(drawdowns_history)
    turnovers_history = torch.stack(turnovers_history)

    loss = -torch.mean(rewards)

    result = {
        "loss": loss,
        "mean_reward": torch.mean(rewards),
        "cum_return": values_history[-1] / initial_value - 1.0,
        "mean_return": torch.mean(portfolio_returns),
        "volatility": torch.std(portfolio_returns, unbiased=False),
        "max_drawdown": torch.max(drawdowns_history),
        "mean_turnover": torch.mean(turnovers_history),
        "weights": weights_history,
        "portfolio_values": values_history,
        "portfolio_returns": portfolio_returns,
        "rewards": rewards,
        "drawdowns": drawdowns_history,
        "turnovers": turnovers_history,
    }

    return result


def evaluate(model, features_df, returns_df):
    model.eval()

    features_tensor = to_tensor(features_df)
    returns_tensor = to_tensor(returns_df)

    with torch.no_grad():
        result = simulate_portfolio(model, features_tensor, returns_tensor)

    return {
        "loss": float(result["loss"]),
        "mean_reward": float(result["mean_reward"]),
        "cum_return": float(result["cum_return"]),
        "mean_return": float(result["mean_return"]),
        "volatility": float(result["volatility"]),
        "max_drawdown": float(result["max_drawdown"]),
        "mean_turnover": float(result["mean_turnover"]),
    }


def save_policy_weights(model, features, returns, asset_names):
    model.eval()

    features_tensor = to_tensor(features)
    returns_tensor = to_tensor(returns)

    with torch.no_grad():
        result = simulate_portfolio(model, features_tensor, returns_tensor)

    weights = result["weights"].detach().cpu().numpy()

    weights_df = pd.DataFrame(
        weights,
        index=features.index,
        columns=[f"w_{asset}" for asset in asset_names],
    )

    weights_df["portfolio_value"] = result["portfolio_values"].detach().cpu().numpy()
    weights_df["portfolio_return"] = result["portfolio_returns"].detach().cpu().numpy()
    weights_df["reward"] = result["rewards"].detach().cpu().numpy()
    weights_df["drawdown"] = result["drawdowns"].detach().cpu().numpy()
    weights_df["turnover"] = result["turnovers"].detach().cpu().numpy()

    weights_df.to_csv(WEIGHTS_FILE)

    return weights_df


# =========================
# Main training loop
# =========================

def main():
    set_seed(SEED)

    features, returns = load_data()
    splits = split_data(features, returns)

    train_features, train_returns = splits["train"]
    val_features, val_returns = splits["val"]
    test_features, test_returns = splits["test"]

    state_dim = train_features.shape[1]
    n_assets = train_returns.shape[1]
    asset_names = list(train_returns.columns)

    print("Datos cargados.")
    print("State dim:", state_dim)
    print("Número de activos:", n_assets)
    print("Activos:", asset_names)
    print("Train:", train_features.index.min().date(), "->", train_features.index.max().date(), train_features.shape)
    print("Validation:", val_features.index.min().date(), "->", val_features.index.max().date(), val_features.shape)
    print("Test:", test_features.index.min().date(), "->", test_features.index.max().date(), test_features.shape)

    model = PolicyNetwork(
        state_dim=state_dim,
        n_assets=n_assets,
        hidden_dim=HIDDEN_DIM,
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_features_tensor = to_tensor(train_features)
    train_returns_tensor = to_tensor(train_returns)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        optimizer.zero_grad()

        train_result = simulate_portfolio(
            model,
            train_features_tensor,
            train_returns_tensor,
        )

        loss = train_result["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_metrics = {
            "loss": float(train_result["loss"]),
            "mean_reward": float(train_result["mean_reward"]),
            "cum_return": float(train_result["cum_return"]),
            "mean_return": float(train_result["mean_return"]),
            "volatility": float(train_result["volatility"]),
            "max_drawdown": float(train_result["max_drawdown"]),
            "mean_turnover": float(train_result["mean_turnover"]),
        }

        val_metrics = evaluate(model, val_features, val_returns)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), MODEL_FILE)

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train loss: {train_metrics['loss']:.6f} | "
                f"Train cum return: {train_metrics['cum_return']:.4f} | "
                f"Train DD: {train_metrics['max_drawdown']:.4f} | "
                f"Val loss: {val_metrics['loss']:.6f} | "
                f"Val cum return: {val_metrics['cum_return']:.4f} | "
                f"Val DD: {val_metrics['max_drawdown']:.4f}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(TRAIN_HISTORY_FILE, index=False)

    print("\nEntrenamiento terminado.")
    print("Mejor modelo guardado en:", MODEL_FILE)
    print("Historial guardado en:", TRAIN_HISTORY_FILE)

    # Cargar mejor modelo
    model.load_state_dict(torch.load(MODEL_FILE))

    print("\nEvaluación final:")
    for split_name, (x_df, r_df) in splits.items():
        metrics = evaluate(model, x_df, r_df)

        print(f"\n{split_name.upper()}")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

    # Guardar pesos para toda la muestra
    weights_df = save_policy_weights(model, features, returns, asset_names)

    print("\nPesos dinámicos guardados en:", WEIGHTS_FILE)
    print("\nÚltimas filas de pesos:")
    print(weights_df.tail())


if __name__ == "__main__":
    main()