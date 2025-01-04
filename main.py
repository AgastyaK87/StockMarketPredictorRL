import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env


# Environment Definition
class PortfolioEnv(gym.Env):
    def __init__(self, portfolio_state, tickers, initial_cash):
        super(PortfolioEnv, self).__init__()
        self.portfolio_state = portfolio_state
        self.tickers = tickers
        self.current_step = 0
        self.prev_portfolio_value = initial_cash
        self.num_assets = len(tickers)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets, self.portfolio_state.shape[2], self.portfolio_state.shape[3]),
            dtype=np.float64,
        )

    def render(self, mode="human"):
        pass

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Extract initial prices and set equal allocation
        initial_prices = self.portfolio_state[:, self.current_step, -1, 0]
        initial_action = np.ones(self.num_assets) / self.num_assets  # Equal allocation

        # Set the initial portfolio value based on prices and allocation
        self.prev_portfolio_value = np.sum(initial_action * initial_prices)
        self.prev_action = initial_action  # Save the initial action

        print(f"Initial Portfolio Value: {self.prev_portfolio_value}")
        print(f"Initial Prices: {initial_prices}")

        # Return the first state
        initial_state = self.portfolio_state[:, self.current_step, :, :]
        return initial_state, {}

    def step(self, action):
        # Normalize the action
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action += 1e-8  # Avoid division by zero
        action /= np.sum(action)

        # Debugging
        print(f"Step {self.current_step}: Action: {action}")

        # Extract "Close" prices for all assets
        current_prices = self.portfolio_state[:, self.current_step, -1, 0]
        print(f"Step {self.current_step}: Current Prices: {current_prices}")

        # Compute reward
        reward = self._compute_reward(action, current_prices)
        print(f"Step {self.current_step}: Reward: {reward}")

        # Update portfolio value
        self.prev_portfolio_value = np.sum(action * current_prices)
        print(f"Step {self.current_step}: Portfolio Value: {self.prev_portfolio_value}")

        # Increment step
        self.current_step += 1

        # Check if terminated (end of data)
        terminated = self.current_step >= self.portfolio_state.shape[1]

        # Truncated (not used in this example, so always False)
        truncated = False

        # Get the next state or return None if terminated
        next_state = self.portfolio_state[:, self.current_step, :, :] if not terminated else None

        return next_state, reward, terminated, truncated, {}

    def _compute_reward(self, action, current_prices):
        portfolio_value = np.sum(action * current_prices)
        print(f"Computed Portfolio Value: {portfolio_value}")
        if self.prev_portfolio_value == 0:
            self.prev_portfolio_value = 1e-8
        self.prev_action = action
        penalty_factor = 0.02
        risk_penalty = np.std(action) * penalty_factor
        reward = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        change_penalty = np.sum(np.abs(action - self.prev_action)) * penalty_factor
        reward -= (change_penalty+risk_penalty)
        print(f"Computed Reward: {reward}")
        if np.isnan(reward):
            raise ValueError("Reward calculation resulted in NaN.")
        return reward


# Data Fetching and Preprocessing
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["RSI"] = compute_rsi(data["Close"])
    data = data.dropna()  # Remove rows with NaN
    return data


def fetch_data_for_tickers(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        all_data.append(data[["Close", "SMA_20", "RSI"]])
    return all_data


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def prepare_state(data, window_size):
    states = []
    for t in range(window_size, len(data)):
        state = data.iloc[t - window_size : t].values
        states.append(state)
    return np.array(states)


def prepare_state_for_tickers(all_data, window_size):
    state_data = []
    for data in all_data:
        rolling_windows = prepare_state(data, window_size)
        state_data.append(rolling_windows)
    return np.array(state_data)


# Main Function
if __name__ == "__main__":
    # Define tickers and fetch data
    tickers = ["NVDA", "GOOG", "SNOW", "PLTR", "AMD", "AMZN", "AAPL", "MSFT", "TSLA"]
    start_date = "2022-10-01"
    end_date = "2024-12-28"
    window_size = 20

    all_data = fetch_data_for_tickers(tickers, start_date, end_date)
    state_data = prepare_state_for_tickers(all_data, window_size)

    print(f"State data shape: {state_data.shape}")

    # Initialize the environment
    env = PortfolioEnv(portfolio_state=state_data, tickers=tickers, initial_cash=5000)

    # Debug environment compatibility
    check_env(env)

    # Train PPO agent
    print("Starting PPO training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    print("PPO training completed.")

    # Save and evaluate
    model.save("ppo_portfolio")
    print("Model saved as 'ppo_portfolio'.")

    # Evaluate performance
    obs, _ = env.reset()  # Extract the observation from the tuple
    done = False
    total_reward = 0
    portfolio_values = [env.prev_portfolio_value]

    # Simulate an episode
    while not done:
        # Pass only the observation to model.predict()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)  # Extract the new observation
        total_reward += reward
        portfolio_values.append(env.prev_portfolio_value)

    # Plot portfolio value over time
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Total Reward: {total_reward}")
