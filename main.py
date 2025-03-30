import os
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env


# Environment Definition
class PortfolioEnv(gym.Env):
    def __init__(self, portfolio_state, tickers, initial_cash, include_cash_in_state=False):
        super(PortfolioEnv, self).__init__()
        self.portfolio_state = portfolio_state
        self.tickers = tickers
        self.current_step = 0
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.num_assets = len(tickers)
        self.shares = np.zeros(self.num_assets)
        self.transaction_cost_pct = 0.001  # 0.1% transaction cost
        self.include_cash_in_state = include_cash_in_state

        # Define action and observation spaces
        # Action space includes allocation for all assets (cash is implicit)
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float64
        )
        
        # Observation space - can include cash position or not
        if include_cash_in_state:
            # Add 1 to include cash position in the observation
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_assets + 1, self.portfolio_state.shape[2], self.portfolio_state.shape[3]),
                dtype=np.float64,
            )
        else:
            # Original observation space without cash
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
        self.cash = self.initial_cash
        self.shares = np.zeros(self.num_assets)
        
        # Extract initial prices
        initial_prices = self.portfolio_state[:, self.current_step, -1, 0]
        
        # Initial action is to hold all cash (no stocks)
        self.prev_action = np.zeros(self.num_assets)
        self.prev_portfolio_value = self.cash
        
        print(f"Initial Cash: ${self.cash:.2f}")
        print(f"Initial Portfolio Value: ${self.prev_portfolio_value:.2f}")
        print(f"Initial Prices: {initial_prices}")

        # Return the first state with cash information
        initial_state = self._get_observation()
        return initial_state, {}
        
    def _get_observation(self):
        # Get market data for current step
        market_data = self.portfolio_state[:, self.current_step, :, :]
        
        if self.include_cash_in_state:
            # Calculate portfolio value
            current_prices = self.portfolio_state[:, self.current_step, -1, 0]
            stock_value = self.shares * current_prices
            total_value = np.sum(stock_value) + self.cash
            
            # Create cash observation (same shape as one asset's data but filled with cash value)
            # We use the same shape as market data for consistency
            cash_data = np.ones((1, market_data.shape[1], market_data.shape[2])) * self.cash / total_value
            
            # Combine market data with cash data
            full_observation = np.vstack([market_data, cash_data])
            
            return full_observation
        else:
            # Return just the market data without cash information
            return market_data

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

        # Calculate current portfolio value before rebalancing
        stock_value = self.shares * current_prices
        old_portfolio_value = np.sum(stock_value) + self.cash
        print(f"Current Portfolio Value (before rebalancing): ${old_portfolio_value:.2f}")
        print(f"Current Cash: ${self.cash:.2f}")
        print(f"Current Shares: {self.shares}")

        # Calculate target allocation in dollars
        target_value_per_asset = action * old_portfolio_value
        
        # Calculate shares to buy/sell
        target_shares = target_value_per_asset / current_prices
        shares_delta = target_shares - self.shares
        
        # Calculate transaction costs
        transaction_cost = np.sum(np.abs(shares_delta * current_prices)) * self.transaction_cost_pct
        
        # Update cash and shares
        self.cash = old_portfolio_value - np.sum(target_shares * current_prices) - transaction_cost
        self.shares = target_shares
        
        # Calculate new portfolio value after rebalancing
        stock_value = self.shares * current_prices
        new_portfolio_value = np.sum(stock_value) + self.cash
        
        # Compute reward
        reward = self._compute_reward(old_portfolio_value, new_portfolio_value, action)
        print(f"Step {self.current_step}: Reward: {reward}")
        print(f"New Portfolio Value (after rebalancing): ${new_portfolio_value:.2f}")
        print(f"New Cash: ${self.cash:.2f}")
        print(f"New Shares: {self.shares}")
        print(f"Transaction Cost: ${transaction_cost:.2f}")

        # Update previous portfolio value and action
        self.prev_portfolio_value = new_portfolio_value
        self.prev_action = action

        # Increment step
        self.current_step += 1

        # Check if terminated (end of data)
        terminated = self.current_step >= self.portfolio_state.shape[1] - 1

        # Truncated (not used in this example, so always False)
        truncated = False

        # Get the next state
        if terminated:
            next_state = None
        else:
            next_state = self._get_observation()

        return next_state, reward, terminated, truncated, {}

    def _compute_reward(self, old_value, new_value, action):
        # Calculate return
        if old_value == 0:
            old_value = 1e-8  # Avoid division by zero
        
        # Base reward is the portfolio return
        reward = (new_value - old_value) / old_value
        
        # Add penalties
        penalty_factor = 0.02
        
        # Risk penalty - penalize concentrated portfolios
        risk_penalty = np.std(action) * penalty_factor
        
        # Change penalty - penalize frequent large changes in allocation
        change_penalty = np.sum(np.abs(action - self.prev_action)) * penalty_factor
        
        # Apply penalties
        reward -= (change_penalty + risk_penalty)
        
        print(f"Base Return: {(new_value - old_value) / old_value:.4f}")
        print(f"Risk Penalty: {risk_penalty:.4f}")
        print(f"Change Penalty: {change_penalty:.4f}")
        print(f"Final Reward: {reward:.4f}")
        
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


def get_user_input_cash():
    """Get user input for initial cash amount"""
    while True:
        try:
            cash_input = input("Enter initial cash amount (default: $10,000): $")
            if not cash_input:
                return 10000  # Default value
            cash = float(cash_input)
            if cash <= 0:
                print("Cash amount must be positive. Please try again.")
                continue
            return cash
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def train_portfolio_model(tickers, start_date, end_date, window_size, initial_cash, include_cash_in_state=False, timesteps=10000):
    """Train a portfolio optimization model"""
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    all_data = fetch_data_for_tickers(tickers, start_date, end_date)
    state_data = prepare_state_for_tickers(all_data, window_size)
    print(f"State data shape: {state_data.shape}")

    # Initialize the environment
    env = PortfolioEnv(
        portfolio_state=state_data, 
        tickers=tickers, 
        initial_cash=initial_cash,
        include_cash_in_state=include_cash_in_state
    )

    # Debug environment compatibility
    check_env(env)

    # Train PPO agent
    print("Starting PPO training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    print("PPO training completed.")

    # Save the model
    model.save("ppo_portfolio")
    print("Model saved as 'ppo_portfolio'.")
    
    return model, env, state_data


def evaluate_model(model, env, tickers, initial_cash):
    """Evaluate a trained model"""
    print(f"\nEvaluating model with ${initial_cash:.2f} initial cash...")
    
    # Reset environment with the specified initial cash
    env.initial_cash = initial_cash
    obs, _ = env.reset()
    
    # Debug information
    print(f"Model observation space: {model.observation_space.shape}")
    print(f"Environment observation space: {env.observation_space.shape}")
    print(f"Actual observation shape: {obs.shape}")
    
    # Track performance
    done = False
    total_reward = 0
    portfolio_values = [env.prev_portfolio_value]
    cash_values = [env.cash]
    stock_values = [np.sum(env.shares * env.portfolio_state[:, env.current_step, -1, 0])]
    allocations = [np.zeros(len(tickers) + 1)]  # +1 for cash
    
    step = 0
    # Simulate an episode
    while not done:
        # Get action from model
        action, _ = model.predict(obs)
        
        # Take step
        obs, reward, done, truncated, _ = env.step(action)
        
        # Track metrics
        total_reward += reward
        portfolio_values.append(env.prev_portfolio_value)
        cash_values.append(env.cash)
        
        if not done:
            stock_value = np.sum(env.shares * env.portfolio_state[:, env.current_step, -1, 0])
        else:
            stock_value = stock_values[-1]  # Use last value if done
            
        stock_values.append(stock_value)
        
        # Calculate allocation percentages including cash
        total_value = env.prev_portfolio_value
        if total_value > 0:
            stock_allocation = (env.shares * env.portfolio_state[:, min(env.current_step, env.portfolio_state.shape[1]-1), -1, 0]) / total_value
            cash_allocation = env.cash / total_value
            current_allocation = np.append(stock_allocation, cash_allocation)
        else:
            current_allocation = np.zeros(len(tickers) + 1)
            
        allocations.append(current_allocation)
        
        step += 1
        if step % 10 == 0:
            print(f"Step {step}: Portfolio Value: ${env.prev_portfolio_value:.2f}")
    
    # Convert to numpy arrays for easier manipulation
    portfolio_values = np.array(portfolio_values)
    cash_values = np.array(cash_values)
    stock_values = np.array(stock_values)
    allocations = np.array(allocations)
    
    # Calculate performance metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Print performance summary
    print("\n===== Performance Summary =====")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Reward (RL metric): {total_reward:.4f}")
    
    # Plot results
    plot_results(portfolio_values, cash_values, stock_values, allocations, tickers)
    
    return portfolio_values, allocations


def plot_results(portfolio_values, cash_values, stock_values, allocations, tickers):
    """Plot portfolio performance and allocations"""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Portfolio Value
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(portfolio_values, 'b-', linewidth=2, label='Total Portfolio')
    ax1.plot(cash_values, 'g--', linewidth=1, label='Cash')
    ax1.plot(stock_values, 'r--', linewidth=1, label='Stocks')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Asset Allocation
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Create stacked area chart for allocations
    x = np.arange(allocations.shape[0])
    y_stack = np.zeros(allocations.shape[0])
    
    # Plot each asset allocation
    for i in range(len(tickers)):
        ax2.fill_between(x, y_stack, y_stack + allocations[:, i], 
                         alpha=0.7, label=tickers[i])
        y_stack += allocations[:, i]
    
    # Add cash allocation
    ax2.fill_between(x, y_stack, y_stack + allocations[:, -1], 
                     alpha=0.7, label='Cash')
    
    ax2.set_title('Portfolio Allocation Over Time')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Allocation (%)')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# Main Function
if __name__ == "__main__":
    # Define tickers and parameters
    tickers = ["NVDA", "GOOG", "SNOW", "PLTR", "AMD", "AMZN", "AAPL", "MSFT", "TSLA"]
    start_date = "2022-10-01"
    end_date = "2024-12-28"
    window_size = 20
    
    # Get user input for initial cash
    initial_cash = get_user_input_cash()
    
    # For now, we'll set include_cash to False to match the original model's expectations
    # This ensures compatibility with the existing model
    include_cash = False
    print("Using standard state representation (without cash in observation).")
    
    # Ask if user wants to train a new model or use existing
    use_existing = input("Use existing model if available? (y/n): ").lower() == 'y'
    
    if use_existing and os.path.exists("ppo_portfolio.zip"):
        print("Loading existing model...")
        # Fetch data for environment
        all_data = fetch_data_for_tickers(tickers, start_date, end_date)
        state_data = prepare_state_for_tickers(all_data, window_size)
        
        # Create environment with same settings as when model was trained
        # For existing models, we assume they were trained without cash in state
        env = PortfolioEnv(
            portfolio_state=state_data, 
            tickers=tickers, 
            initial_cash=initial_cash,
            include_cash_in_state=include_cash
        )
        
        # Load model
        model = PPO.load("ppo_portfolio")
        print("Model loaded successfully.")
    else:
        # Train new model
        model, env, state_data = train_portfolio_model(
            tickers, start_date, end_date, window_size, initial_cash, include_cash
        )
    
    # Evaluate model
    evaluate_model(model, env, tickers, initial_cash)
    
    # Allow testing with different cash amounts
    while True:
        test_another = input("\nTest with a different cash amount? (y/n): ").lower()
        if test_another != 'y':
            break
            
        test_cash = get_user_input_cash()
        evaluate_model(model, env, tickers, test_cash)
