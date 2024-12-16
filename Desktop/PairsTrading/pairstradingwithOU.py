import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from itertools import combinations
import os
import logging
from concurrent.futures import ProcessPoolExecutor  # Import for multiprocessing
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize
from typing import List, Optional


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
config = {
    'risk_free_rate': 0.04,
    'trading_days': 252,
    'commission_rate': 0.004,
    'tax_rate': 0.075,  # Added tax rate for profitable trades
    'p_value_threshold': 0.05,
    'num_simulations': 100,
    'walk_forward_window': 252,
    'min_window': 5,
    'max_window' : 30
}

# Min and Max window params is for mean reverting process convergence and will determine the rolling window

def process_stock_data(
    stock_symbols: List[str], 
    data_directory: str = '/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/', 
    fill_method: Optional[str] = 'ffill'
) -> pd.DataFrame:
    """
    Process stock data ensuring precise date alignment and order.
    
    Args:
        stock_symbols (List[str]): List of stock symbols to process
        data_directory (str): Path to directory containing stock CSV files
        fill_method (Optional[str]): Method to fill missing values
    
    Returns:
        pd.DataFrame: Aligned stock closing prices
    """
    logging.basicConfig(level=logging.INFO)
    
    if len(stock_symbols) < 2:
        logging.error("At least two stock symbols required")
        return pd.DataFrame()

    # Store processed dataframes
    stock_dataframes = {}
    
    for symbol in stock_symbols:
        try:
            # Read stock data
            df = pd.read_csv(
                f'{data_directory}{symbol}.csv', 
                parse_dates=['date'], 
                index_col='date'
            )
            
            # Clean and prepare data
            df.index = pd.to_datetime(df.index).date
            df = df.loc[~df.index.duplicated(keep='first')]
            df = df[df['close'] != 0].dropna(subset=['close'])
            
            stock_dataframes[symbol] = df[['close']]
        
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if len(stock_dataframes) < 2:
        logging.error("Insufficient valid stock data")
        return pd.DataFrame()

    # Find the latest start date and earliest end date
    start_date = max(df.index.min() for df in stock_dataframes.values())
    end_date = min(df.index.max() for df in stock_dataframes.values())

    # Combine data with precise filtering
    combined_df = pd.DataFrame(index=pd.date_range(start_date, end_date))
    for symbol, df in stock_dataframes.items():
        combined_df[symbol] = df.loc[start_date:end_date, 'close']

    # Apply fill method and remove NaN rows
    if fill_method:
        combined_df = combined_df.fillna(method=fill_method)
    combined_df.dropna(inplace=True)

    # Save with both stock names
    output_filename = f'{stock_symbols[0]}_{stock_symbols[1]}_combined_data.csv'
    #combined_df.to_csv(output_filename)
    
    logging.info(f"Processed data for {len(stock_symbols)} stocks")
    return combined_df


def validate_pair(stock_a, stock_b, data, min_periods=30, min_window=config['min_window'], max_window=config['max_window']):
    """
    Validates a pair of stocks for trading based on several criteria including half-life constraints.
    
    Parameters:
    -----------
    stock_a, stock_b : str
        Stock symbols to validate
    data : pd.DataFrame
        DataFrame containing price data for both stocks
    min_periods : int
        Minimum number of periods required for validation
    min_window : int
        Minimum allowed window size for half-life
    max_window : int
        Maximum allowed window size for half-life
        
    Returns:
    --------
    tuple : (is_valid, reason, half_life, rolling_window)
        - is_valid: boolean indicating if pair is valid
        - reason: string explaining validation result
        - half_life: calculated half-life if valid, None if invalid
        - rolling_window: suggested rolling window size if valid, None if invalid
    """
    try:
        # Check for sufficient data
        if len(data) < min_periods:
            return False, "Insufficient data points", None, None

        # Check for missing values
        if data[stock_a].isnull().any() or data[stock_b].isnull().any():
            return False, "Contains missing values", None, None

        # Check for zero or negative prices
        if (data[stock_a] <= 0).any() or (data[stock_b] <= 0).any():
            return False, "Contains zero or negative prices", None, None

        # Calculate spread
        hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])
        spread = data[stock_a] - hedge_ratio * data[stock_b]

        # Calculate half-life
        theta, mu, sigma = OU(spread)
        half_life = half_life_calc(theta)

        # Validate half-life is a valid number
        if half_life <= 0 or np.isnan(half_life) or np.isinf(half_life):
            return False, "Invalid half-life value", None, None

        # Check if half-life is within the allowed window range
        if half_life < min_window:
            return False, f"Half-life ({half_life:.2f}) is less than minimum window ({min_window})", None, None
            
        if half_life > max_window:
            return False, f"Half-life ({half_life:.2f}) exceeds maximum window ({max_window})", None, None

        # Calculate suggested rolling window based on half-life
        rolling_window = int(max(min_window, min(half_life, max_window)))
        
        return True, "Valid pair", half_life, rolling_window

    except Exception as e:
        return False, f"Error during validation: {str(e)}", None, None

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate, trading_days=config['trading_days']):
    excess_returns = returns - (risk_free_rate / trading_days)
    annualized_excess_returns = excess_returns.mean() * trading_days
    annualized_volatility = excess_returns.std() * np.sqrt(trading_days)
    
    # Avoid division by zero if volatility is zero
    if annualized_volatility == 0:
        return 0

    sharpe_ratio = annualized_excess_returns / annualized_volatility
    return sharpe_ratio


# Function to calculate Sortino ratio
def calculate_sortino_ratio(returns, risk_free_rate=config['risk_free_rate'], trading_days=config['trading_days']):
    downside_returns = returns[returns < risk_free_rate / trading_days]
    expected_return = returns.mean() * trading_days
    downside_deviation = np.sqrt((downside_returns**2).mean()) * trading_days
    return (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

# Function to calculate drawdown
def calculate_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = drawdown.max()
    return max_drawdown, drawdown


# Function to calculate hedge ratio
def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio

# Calculate OU params
def OU(spread):
    """
    Fit an Ornstein-Uhlenbeck process to the given spread data.
    Returns estimated parameters: theta, mu, csigma.
    """
    # Define the OU process objective function
    def ou_objective(params):
        theta, mu, sigma = params
        dt = 1  # Daily data, so time step is 1 day
        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        
        # OU model: dS = theta * (mu - S) * dt + sigma * dW
        predicted_diff = theta * (mu - spread_lag) * dt
        residual = spread_diff - predicted_diff
        
        # Minimize the squared error (residuals)
        return np.sum(residual**2)
    
    # Initial guess for the parameters [theta, mu, sigma]
    initial_guess = [0.5, spread.mean(), spread.std()]
    bounds = [(1e-6, None), (None, None), (1e-6, None)]  # Avoid zero for theta and sigm
    
    # Minimize the objective function to estimate the parameters
    result = minimize(ou_objective, initial_guess, bounds=bounds)
    
    # Extract the fitted parameters
    theta, mu, sigma = result.x
    return theta, mu, sigma

def half_life_calc(theta):
    return np.log(2)/theta


# Function to calculate half-life only once
def get_half_life(stock_a, stock_b, data):
    """
    Fit the OU and get the value of theta
    """
    spread = data[stock_a] - calculate_hedge_ratio(data[stock_a], data[stock_b]) * data[stock_b]
    theta, _, _ = OU(spread)
    if not np.isfinite(theta) or theta <= 0:
        logging.warning(f"Skipping pair {stock_a}-{stock_b} due to invalid theta.")
        return config['min_window']  # Return None for invalid half-life
    half_life = np.log(2) / theta
    return half_life

    
def rolling_mean(arr, window):
    # Compute rolling mean, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].mean() for i in range(len(arr))])

def rolling_std(arr, window):
    # Compute rolling standard deviation, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].std() for i in range(len(arr))])
    
def backtest_stat_arbitrage(hedge_ratio, data, stock_a, stock_b, z_score, half_life, initial_capital=10000, 
                          z_entry_threshold_a=-2.0, z_entry_threshold_b=2.0, z_exit_threshold=0.0, 
                          min_holding_period=3, commission_rate=0.001, stop_loss_threshold=0.05, 
                          cooldown_period=1, min_window=config['min_window'], max_window=config['max_window']):
    """
    Backtest the statistical arbitrage strategy.
    """

    positions_a = pd.Series(0, index=data.index)
    positions_b = pd.Series(0, index=data.index)
    capital = initial_capital

    holding_counter_a = 0
    holding_counter_b = 0
    in_position_a = False
    in_position_b = False
    entry_price_a = 0
    entry_price_b = 0

    cooldown_counter_a = 0
    cooldown_counter_b = 0

    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0

    for i in range(1, len(data)):
        # --- Cooldown Logic ---
        if cooldown_counter_a > 0:
            cooldown_counter_a -= 1
        if cooldown_counter_b > 0:
            cooldown_counter_b -= 1

        # --- Entry Conditions ---
        if not in_position_a and z_score.iloc[i] <= z_entry_threshold_a and cooldown_counter_a == 0:
            positions_a.iloc[i] = 1  # Go long on stock_a
            entry_price_a = data[stock_a].iloc[i]  # Record the entry price of stock_a
            in_position_a = True
            holding_counter_a = 1
            capital -= entry_price_a * (1 + commission_rate)  # Adjust capital for commission
            total_trades += 1

        if not in_position_b and z_score.iloc[i] >= z_entry_threshold_b and cooldown_counter_b == 0:
            positions_b.iloc[i] = 1 * abs(hedge_ratio)# Go long on stock_b
            entry_price_b = data[stock_b].iloc[i]  # Record the entry price of stock_b
            in_position_b = True
            holding_counter_b = 1
            capital -= entry_price_b * (1 + commission_rate)  # Adjust capital for commission
            total_trades += 1

        # --- Exit for stock_a ---
        if in_position_a:
            holding_counter_a += 1
            current_loss_a = (data[stock_a].iloc[i] - entry_price_a) / entry_price_a

            # Exit on stop-loss or if z-score has reverted beyond exit threshold for stock_a
            if (holding_counter_a >= min_holding_period and 
                (current_loss_a < -stop_loss_threshold or z_score.iloc[i] >= z_exit_threshold or holding_counter_a > half_life)):
                positions_a.iloc[i] = -positions_a.iloc[i-1]  # Exit position for stock_a
                exit_price_a = data[stock_a].iloc[i]
                trade_profit_a = (exit_price_a - entry_price_a)  # Calculate profit without shares
                if trade_profit_a > 0:
                    winning_trades += 1
                    trade_profit_a_after_tax = trade_profit_a * (1 - config['tax_rate'])  # Apply tax
                    total_profit += trade_profit_a_after_tax
                else:
                    losing_trades += 1
                    total_loss += abs(trade_profit_a)
                in_position_a = False
                holding_counter_a = 0
                cooldown_counter_a = cooldown_period  # Reset cooldown for stock_a
                capital += exit_price_a * (1 - commission_rate)  # Adjust capital for commission

        # --- Exit for stock_b ---
        if in_position_b:
            holding_counter_b += 1
            current_loss_b = (data[stock_b].iloc[i] - entry_price_b) / entry_price_b

            # Exit on stop-loss or if z-score has reverted beyond exit threshold for stock_b
            if (holding_counter_b >= min_holding_period and 
                (current_loss_b < -stop_loss_threshold or z_score.iloc[i] <= z_exit_threshold or holding_counter_b > half_life)):
                positions_b.iloc[i] = -positions_b.iloc[i-1]  # Exit position for stock_b
                exit_price_b = data[stock_b].iloc[i]
                trade_profit_b = (exit_price_b - entry_price_b)  # Calculate profit without shares
                if trade_profit_b > 0:
                    winning_trades += 1
                    trade_profit_b_after_tax = trade_profit_b * (1 - config['tax_rate'])  # Apply tax
                    total_profit += trade_profit_b_after_tax
                else:
                    losing_trades += 1
                    total_loss += abs(trade_profit_b)
                in_position_b = False
                holding_counter_b = 0
                cooldown_counter_b = cooldown_period  # Reset cooldown for stock_b
                capital += exit_price_b * (1 - commission_rate)  # Adjust capital for commission

    # Calculate returns for stock_a and stock_b
    stock_returns_a = data[stock_a].pct_change().fillna(0)
    stock_returns_b = data[stock_b].pct_change().fillna(0)

    # Combine strategy returns from both stocks
    strategy_returns_a = (positions_a.shift(1) * stock_returns_a).fillna(0)
    strategy_returns_b = (positions_b.shift(1) * stock_returns_b).fillna(0)
    strategy_returns = strategy_returns_a + strategy_returns_b

    # Adjust returns for commissions on trades
    trades_a = positions_a.diff().fillna(0)
    trades_b = positions_b.diff().fillna(0)
    commission_costs = (trades_a.abs() + trades_b.abs()) * commission_rate
    strategy_returns -= commission_costs.shift(1).fillna(0)

    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod() * initial_capital

    # Ensure we exit positions on the last day if still in position
    if in_position_a:
        positions_a.iloc[-1] = -positions_a.shift(1).iloc[-1]  # Exit position for stock_a
        strategy_returns_a.iloc[-1] = -positions_a.shift(1).iloc[-1] * stock_returns_a.iloc[-1]
    if in_position_b:
        positions_b.iloc[-1] = -positions_b.shift(1).iloc[-1]  # Exit position for stock_b
        strategy_returns_b.iloc[-1] = -positions_b.shift(1).iloc[-1] * stock_returns_b.iloc[-1]

    # Recalculate combined strategy returns after exiting
    strategy_returns = strategy_returns_a + strategy_returns_b

    # Calculate performance metrics: drawdown, Sharpe ratio, Sortino ratio
    max_drawdown, _ = calculate_drawdown(cumulative_returns.pct_change().dropna())
    sharpe_ratio = calculate_sharpe_ratio(cumulative_returns.pct_change().dropna(), config['risk_free_rate'])
    sortino_ratio = calculate_sortino_ratio(cumulative_returns.pct_change().dropna())

    # Calculate additional trade metrics
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    average_profit = total_profit / winning_trades if winning_trades > 0 else 0
    average_loss = total_loss / losing_trades if losing_trades > 0 else 0

    return (cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown, 
            total_trades, win_rate, profit_factor, average_profit, average_loss)




def monte_carlo_simulation(stock_a, stock_b, data, num_simulations=config['num_simulations'], 
                         walk_forward_window=config['walk_forward_window'],
                         min_window=config['min_window'], max_window=config['max_window']):
    """
    Modified Monte Carlo simulation
    """
    
    # Proceed with simulation only for valid pairs
    results = []
    for sim in range(num_simulations):
        # Randomize parameters for each simulation
        z_entry_threshold_a = np.random.uniform(-3.0, -1.0)  # For stock_a (long)
        z_entry_threshold_b = np.random.uniform(1.0, 3.0)    # For stock_b (long)
        z_exit_threshold = np.random.uniform(-0.5, 0.5)      # Exit threshold

        print('started', z_entry_threshold_a, z_entry_threshold_b, z_exit_threshold)
        
        # Iterate over different stop-loss thresholds
        for stop_loss_threshold in np.arange(0.05, 0.21, 0.05):
            # Divide data into training and testing sets
            train_start = 0
            train_end = len(data) - walk_forward_window
            test_start = train_end
            test_end = len(data)

            print('done framing the datas')

            while test_end <= len(data):
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]
                train_half_life = get_half_life(stock_a, stock_b, train_data)
                test_half_life = get_half_life(stock_a, stock_b, test_data)

                if train_half_life is None or train_half_life < config['min_window']:
                    train_half_life = config['min_window']
                #print('train half life is', train_half_life)
                #print('test half life is ', test_half_life )
                if test_half_life is None or test_half_life < config['min_window']:
                    test_half_life = config['min_window']
                
                # Calculate the hedge ratio
                hedge_ratio_train = calculate_hedge_ratio(train_data[stock_a], train_data[stock_b])
                hedge_ratio_test = calculate_hedge_ratio(test_data[stock_a], test_data[stock_b])

                # Calculate spread
                train_data_spread = train_data[stock_a] - hedge_ratio_train * train_data[stock_b]
                test_data_spread = test_data[stock_a] - hedge_ratio_test * test_data[stock_b]

                # Calculate z-scores
                train_spread_mean = train_data_spread.rolling(window=int(train_half_life), min_periods=int(train_half_life)).mean()
                train_spread_std = train_data_spread.rolling(window=int(train_half_life), min_periods=int(train_half_life)).std()
                z_score_train = (train_data_spread - train_spread_mean) / train_spread_std
                test_spread_mean = test_data_spread.rolling(window=int(test_half_life), min_periods=int(test_half_life)).mean()
                test_spread_std = test_data_spread.rolling(window=int(test_half_life), min_periods=int(test_half_life)).std()
                z_score_test = (test_data_spread - test_spread_mean) / test_spread_std


                # Replace infinite values and NaN in z-scores
                z_score_train = z_score_train.replace([np.inf, -np.inf], np.nan)
                z_score_train = z_score_train.fillna(0)
                z_score_test = z_score_test.replace([np.inf, -np.inf], np.nan)
                z_score_test = z_score_test.fillna(0)

                print('training set iteration')
                # Backtest on training data
                (cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown,
                 total_trades, win_rate, profit_factor, average_profit, average_loss) = backtest_stat_arbitrage(hedge_ratio_train,
                    train_data, stock_a, stock_b, z_score_train,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    min_window=min_window,
                    max_window=max_window,
                    half_life=train_half_life
                )
                print('cum returns', cumulative_returns)

                # Evaluate on testing data
                (test_cumulative_returns, _, _, _, _, _, _, _, _) = backtest_stat_arbitrage(hedge_ratio_test,
                    test_data, stock_a, stock_b, z_score_test,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    min_window=min_window,
                    max_window=max_window,
                    half_life=test_half_life
                )

                risk_adjusted_return = sharpe_ratio if max_drawdown == 0 else sharpe_ratio / (1 + max_drawdown)


                results.append({
                    'cumulative_returns': cumulative_returns.iloc[-1],
                    'test_cumulative_returns': test_cumulative_returns.iloc[-1],
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'risk_adjusted_return': risk_adjusted_return,
                    'z_entry_threshold_a': z_entry_threshold_a,
                    'z_entry_threshold_b': z_entry_threshold_b,
                    'z_exit_threshold': z_exit_threshold,
                    'stop_loss_threshold': stop_loss_threshold,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_profit': average_profit,
                    'average_loss': average_loss,
                })

                # Update training and testing windows
                train_start += walk_forward_window
                train_end += walk_forward_window
                test_start += walk_forward_window
                test_end += walk_forward_window
                print('next iteration')

    return pd.DataFrame(results)



# Getting the stock list from the datas folder
# Function to get all filenames from the 'datas' folder
def get_stock_filenames(data_folder='/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/'):
    filenames = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    stock_symbols = [f.replace('.csv', '') for f in filenames]
    return stock_symbols

# Cointegrated assets
# Load cointegrated pairs
def load_cointegrated_pairs(filepath='/Users/icarus/Desktop/QuantatitiveTradingStrategies/cointegration_results.csv'):
    return pd.read_csv(filepath)

# Change the current working directory
os.chdir('/Users/icarus/Desktop/QuantatitiveTradingStrategies/Pairs Trading')

# Create directories for results and plots
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Check for existing validation results
VALIDATION_RESULTS_PATH = 'results/pair_validation_results.csv'

if __name__ == "__main__":
    all_final_results = []

    # Load cointegrated pairs from file
    cointegrated_pairs = load_cointegrated_pairs()

    # Check if validation results already exist
    if os.path.exists(VALIDATION_RESULTS_PATH):
        logging.info("Loading existing validation results.")
        validation_results = pd.read_csv(VALIDATION_RESULTS_PATH)

        # Filter valid and rejected pairs
        valid_pairs = validation_results[validation_results['status'] == 'valid'][['stock_a', 'stock_b', 'reason/half_life']].values.tolist()
        rejected_pairs = validation_results[validation_results['status'] == 'rejected'][['stock_a', 'stock_b', 'reason/half_life']].values.tolist()

    else:
        logging.info("Validation results not found. Starting validation process.")
        valid_pairs = []
        rejected_pairs = []

        # First pass: validate all pairs
        for _, row in cointegrated_pairs.iterrows():
            stock_a, stock_b = row['stock_a'], row['stock_b']
            data = process_stock_data([stock_a, stock_b], fill_method='ffill')

            is_valid, reason, half_life, rolling_window = validate_pair(stock_a, stock_b, data)

            # Check if half_life is valid (not None) and meets conditions
            if is_valid and half_life and rolling_window is not None:
                rolling_window = int(max(1, min(max(half_life, 10), 50)))

                if rolling_window < 1:
                    # Exclude pair if rolling_window is invalid
                    rejected_pairs.append((stock_a, stock_b, "Invalid rolling window"))
                    logging.warning(f"Rejected pair: {stock_a}-{stock_b} (Invalid rolling window)")
                    continue

                valid_pairs.append((stock_a, stock_b, half_life))
                logging.info(f"Validated pair: {stock_a}-{stock_b} (Half-life: {half_life})")
            else:
                # Exclude pair if half_life is None or other validation issues
                reason = reason or "None half-life"
                rejected_pairs.append((stock_a, stock_b, reason))
                logging.warning(f"Rejected pair: {stock_a}-{stock_b} ({reason})")

        # Save validation results to file
        validation_results = pd.DataFrame({
            'stock_a': [p[0] for p in valid_pairs + rejected_pairs],
            'stock_b': [p[1] for p in valid_pairs + rejected_pairs],
            'status': ['valid' for _ in valid_pairs] + ['rejected' for _ in rejected_pairs],
            'reason/half_life': [p[2] for p in valid_pairs + rejected_pairs]
        })
        os.makedirs(os.path.dirname(VALIDATION_RESULTS_PATH), exist_ok=True)
        validation_results.to_csv(VALIDATION_RESULTS_PATH, index=False)
        logging.info(f"Validation results saved to {VALIDATION_RESULTS_PATH}.")

    # Proceed with analysis only for valid pairs
    if valid_pairs:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    monte_carlo_simulation, 
                    pair[0], pair[1], 
                    process_stock_data([pair[0], pair[1]], fill_method='ffill')
                ): pair for pair in valid_pairs
            }

            for future in futures:
                stock_a, stock_b, half_life = futures[future]
                logging.info(f"Running Monte Carlo simulations for pair: {stock_a}, {stock_b}")

                simulations = future.result()

                if simulations.empty:
                    logging.warning(f"No simulation results for pair: {stock_a}, {stock_b}")
                    continue

                best_result = simulations.loc[simulations['risk_adjusted_return'].idxmax()]

                # Run backtest with best parameters using combined_df for the current pair
                cumulative_returns = best_result['cumulative_returns'],
                sharpe_ratio=best_result['sharpe_ratio'],
                sortino_ratio=best_result['sortino_ratio'],
                max_drawdown=best_result['max_drawdown'],
                total_trades=best_result['total_trades'],
                win_rate=best_result['win_rate']
                profit_factor=best_result['profit_factor'],
                average_profit=best_result['average_profit'],
                average_loss=best_result['average_loss']

                all_final_results.append({
                    'Stock A': stock_a,
                    'Stock B': stock_b,
                    'Z Entry Threshold A': best_result['z_entry_threshold_a'],
                    'Z Entry Threshold B': best_result['z_entry_threshold_b'],
                    'Z Exit Threshold': best_result['z_exit_threshold'],
                    'Stop Loss Threshold': best_result['stop_loss_threshold'],
                    'Final Cumulative Returns': cumulative_returns,
                    'Final Sharpe Ratio': sharpe_ratio,
                    'Final Sortino Ratio': sortino_ratio,
                    'Final Max Drawdown': max_drawdown,
                    'Cumulative Returns Series': cumulative_returns,
                    'Total Trades': total_trades,
                    'Win Rate': win_rate,
                    'Profit Factor': profit_factor,
                    'Average Profit': average_profit,
                    'Average Loss': average_loss
                })


    # Save and plot results as before
    if all_final_results:
        results_df = pd.DataFrame(all_final_results)  # Convert list of dicts to DataFrame

        if 'Cumulative Returns Series' in results_df.columns:
            cumulative_returns_df = pd.DataFrame(results_df['Cumulative Returns Series'].tolist())
            cumulative_returns_df.columns = [f'Timestep {i+1}' for i in range(cumulative_returns_df.shape[1])]

            final_results_df = pd.concat([results_df.drop(columns='Cumulative Returns Series'), cumulative_returns_df], axis=1)

            # Save results to CSV
            csv_filename = 'results/cointegrated_pairs_results.csv'
            final_results_df.to_csv(csv_filename, index=False)
            logging.info(f"Results saved to {csv_filename}")

            # Plot cumulative returns series
            plt.figure(figsize=(12, 6))
            for index, row in final_results_df.iterrows():
                cumulative_returns_series = row[['Timestep 1'] + [f'Timestep {i+1}' for i in range(1, cumulative_returns_df.shape[1])]].values
                cumulative_returns_series = [float(value) for value in cumulative_returns_series]

                # Ensure the date index is in datetime format
                date_index = pd.to_datetime(process_stock_data([row['Stock A'], row['Stock B']], fill_method='ffill').index[:len(cumulative_returns_series)])

                plt.plot(date_index, cumulative_returns_series, label=f'{row["Stock A"]} & {row["Stock B"]}')

            plt.title('Cumulative Returns of Cointegrated Pairs')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_filename = 'plots/cumulative_returns_cointegrated_pairs.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logging.info(f"Cumulative returns plot saved to {plot_filename}")
            plt.close()

        else:
            logging.warning("'Cumulative Returns Series' not found in results.")
    else:
        logging.error("No valid stock data available for analysis.")
