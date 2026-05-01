from pathlib import Path
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")

FEATURES_FILE = PROCESSED_DIR / "features_weekly_normalized.csv"
RETURNS_FILE = PROCESSED_DIR / "returns_weekly_aligned.csv"


class PortfolioEnv:
    """
    Entorno simple de asignación dinámica de activos para DRL.

    Estado:
        s_t = features normalizadas en t

    Acción:
        raw_action -> se transforma a pesos con softmax
        w_t >= 0 y sum(w_t) = 1

    Reward:
        reward_t =
            portfolio_return
            - lambda_vol * rolling_vol_proxy
            - lambda_dd * portfolio_drawdown
            - lambda_to * turnover

    Nota:
        Este entorno no usa gymnasium para mantenerlo simple y controlable.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        start_date=None,
        end_date=None,
        lambda_vol=0.10,
        lambda_dd=0.20,
        lambda_to=0.05,
        initial_value=1.0,
        vol_window=4,
    ):
        self.features = features.copy()
        self.returns = returns.copy()

        common_index = self.features.index.intersection(self.returns.index)
        self.features = self.features.loc[common_index]
        self.returns = self.returns.loc[common_index]

        if start_date is not None:
            self.features = self.features.loc[self.features.index >= pd.to_datetime(start_date)]
            self.returns = self.returns.loc[self.returns.index >= pd.to_datetime(start_date)]

        if end_date is not None:
            self.features = self.features.loc[self.features.index <= pd.to_datetime(end_date)]
            self.returns = self.returns.loc[self.returns.index <= pd.to_datetime(end_date)]

        if len(self.features) < vol_window + 2:
            raise ValueError("Muy pocas observaciones para crear el entorno.")

        self.asset_names = list(self.returns.columns)
        self.n_assets = len(self.asset_names)
        self.state_dim = self.features.shape[1]

        self.lambda_vol = lambda_vol
        self.lambda_dd = lambda_dd
        self.lambda_to = lambda_to
        self.initial_value = initial_value
        self.vol_window = vol_window

        self.reset()

    @staticmethod
    def softmax(x):
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def reset(self):
        self.t = 0
        self.portfolio_value = self.initial_value
        self.peak_value = self.initial_value
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns_history = []

        return self._get_state()

    def _get_state(self):
        return self.features.iloc[self.t].values.astype(np.float32)

    def step(self, action):
        weights = self.softmax(action)

        asset_returns = self.returns.iloc[self.t].values.astype(np.float64)

        portfolio_return = float(np.dot(weights, asset_returns))

        self.portfolio_value *= float(np.exp(portfolio_return))
        self.peak_value = max(self.peak_value, self.portfolio_value)

        portfolio_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        turnover = float(np.sum(np.abs(weights - self.prev_weights)))

        self.portfolio_returns_history.append(portfolio_return)

        if len(self.portfolio_returns_history) >= self.vol_window:
            rolling_vol = float(np.std(self.portfolio_returns_history[-self.vol_window:]))
        else:
            rolling_vol = 0.0

        reward = (
            portfolio_return
            - self.lambda_vol * rolling_vol
            - self.lambda_dd * portfolio_drawdown
            - self.lambda_to * turnover
        )

        self.prev_weights = weights.copy()

        self.t += 1
        done = self.t >= len(self.features)

        next_state = None if done else self._get_state()

        info = {
            "date": self.features.index[self.t - 1],
            "weights": weights,
            "asset_returns": asset_returns,
            "portfolio_return": portfolio_return,
            "portfolio_value": self.portfolio_value,
            "drawdown": portfolio_drawdown,
            "turnover": turnover,
            "rolling_vol": rolling_vol,
            "reward": reward,
        }

        return next_state, float(reward), done, info


def load_environment(start_date=None, end_date=None, **kwargs):
    features = pd.read_csv(FEATURES_FILE, index_col="date", parse_dates=True)
    returns = pd.read_csv(RETURNS_FILE, index_col="date", parse_dates=True)

    env = PortfolioEnv(
        features=features,
        returns=returns,
        start_date=start_date,
        end_date=end_date,
        **kwargs,
    )

    return env


if __name__ == "__main__":
    env = load_environment(start_date="2015-03-27", end_date="2021-12-31")

    state = env.reset()

    print("Entorno creado correctamente.")
    print("State dim:", env.state_dim)
    print("Número de activos:", env.n_assets)
    print("Activos:", env.asset_names)
    print("Primer estado shape:", state.shape)

    random_action = np.random.randn(env.n_assets)

    next_state, reward, done, info = env.step(random_action)

    print("\nPrueba de un step:")
    print("Reward:", reward)
    print("Done:", done)
    print("Fecha:", info["date"])
    print("Pesos:", np.round(info["weights"], 4))
    print("Suma pesos:", round(info["weights"].sum(), 6))
    print("Portfolio return:", round(info["portfolio_return"], 6))
    print("Portfolio value:", round(info["portfolio_value"], 6))
    print("Drawdown:", round(info["drawdown"], 6))
    print("Turnover:", round(info["turnover"], 6))