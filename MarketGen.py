import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MarketGenerator:
    def __init__(self, s0: float = 100.0, n_steps: int = 252, n_paths: int = 10):
        self.s0 = s0
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = 1 / n_steps

    def random_walk(self, mu: float = 0.0, sigma: float = 1.0, seed: int = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        paths = np.zeros((self.n_steps + 1, self.n_paths))
        paths[0] = self.s0

        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(self.n_paths)
            paths[t] = paths[t - 1] + mu * self.dt + sigma * np.sqrt(self.dt) * z

        return pd.DataFrame(paths, columns=[f"RW_{i}" for i in range(self.n_paths)])


    def gbm(self, mu: float, sigma: float, seed: int = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        paths = np.zeros((self.n_steps + 1, self.n_paths))
        paths[0] = self.s0

        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(self.n_paths)
            paths[t] = paths[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * z
            )

        return pd.DataFrame(paths, columns=[f"GBM_{i}" for i in range(self.n_paths)])

    def ornstein_uhlenbeck(self, mu: float, theta: float, sigma: float, seed: int = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        X = np.zeros((self.n_steps + 1, self.n_paths))
        X[0] = self.s0

        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(self.n_paths)
            X[t] = X[t - 1] + theta * (mu - X[t - 1]) * self.dt + sigma * np.sqrt(self.dt) * z

        return pd.DataFrame(X, columns=[f"OU_{i}" for i in range(self.n_paths)])

    def jump_diffusion(self, mu: float, sigma: float, lamb: float, mu_j: float, sigma_j: float, seed: int = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        paths = np.zeros((self.n_steps + 1, self.n_paths))
        paths[0] = self.s0

        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(self.n_paths)
            jumps = np.random.poisson(lamb * self.dt, self.n_paths)
            jump_sizes = np.exp(mu_j + sigma_j * np.random.standard_normal(self.n_paths)) - 1
            paths[t] = paths[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * z
            ) * (1 + jumps * jump_sizes)

        return pd.DataFrame(paths, columns=[f"JD_{i}" for i in range(self.n_paths)])

    def heston(self, mu: float, v0: float, kappa: float, theta: float, xi: float, rho: float, seed: int = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((self.n_steps + 1, self.n_paths))
        v = np.zeros((self.n_steps + 1, self.n_paths))
        S[0], v[0] = self.s0, v0

        for t in range(1, self.n_steps + 1):
            z1 = np.random.standard_normal(self.n_paths)
            z2 = np.random.standard_normal(self.n_paths)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2

            v[t] = np.abs(v[t - 1] + kappa * (theta - v[t - 1]) * self.dt + xi * np.sqrt(v[t - 1] * self.dt) * z2)
            S[t] = S[t - 1] * np.exp((mu - 0.5 * v[t]) * self.dt + np.sqrt(v[t] * self.dt) * z1)

        return pd.DataFrame(S, columns=[f"Heston_{i}" for i in range(self.n_paths)])

    @staticmethod
    def plot(df: pd.DataFrame):
        plt.style.use("seaborn-v0_8-pastel")
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        mean_path = df.mean(axis=1)
        std_path = df.std(axis=1)
        final_prices = df.iloc[-1]

        axes[0].plot(mean_path, color="blue", lw=2, label="Mean Path")
        axes[0].fill_between(df.index,
                             mean_path - std_path,
                             mean_path + std_path,
                             color="blue", alpha=0.2, label="Spread")
        

        max_col = final_prices.idxmax()
        min_col = final_prices.idxmin()
        axes[0].plot(df[max_col], color="green", lw=1, label="Max Path")
        axes[0].plot(df[min_col], color="red", lw=1, label="Min Path")
        axes[0].set_title("Simulated Market Paths")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].hist(final_prices, bins=20, alpha=0.7, edgecolor="black")
        axes[1].axvline(np.mean(final_prices), color="blue", linestyle="--", label="Mean Value")
        axes[1].set_title("Distribution of Final Values")
        axes[1].set_xlabel("Final Value")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("market_simulation.png", dpi = 400)
        plt.show()
