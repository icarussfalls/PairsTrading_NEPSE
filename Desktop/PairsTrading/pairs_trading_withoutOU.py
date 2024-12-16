import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from itertools import combinations
import os
import logging
from concurrent.futures import ProcessPoolExecutor  # Import for multiprocessing
from statsmodels.regression.linear_model import OLS


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
config = {
    'risk_free_rate': 0.05,
    'trading_days': 252,
    'commission_rate': 0.004,
    'tax_rate': 0.075,  # Added tax rate for profitable trades
    'p_value_threshold': 0.05,
    'num_simulations': 1000,
    'walk_forward_window': 100,
    'min_window': 5,
    'max_window' : 30
}

# Min and Max window params is for mean reverting process convergence and will determine the rolling window

# Function to process stock data
def process_stock_data(stock_symbols, fill_method='ffill'):
    aligned_data = {}
    
    for symbol in stock_symbols:
        try:
            data = pd.read_csv(f'/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/{symbol}.csv')
            # Parse the dates and handle invalid dates
            #data['Date'] = pd.to_datetime(data['f_date'], errors='raise')
            #data = data.drop_duplicates(subset='Date')
            #data.sort_values(by='Date', ascending=True, inplace=True)

            # Set the Date as the index for alignment
            #data.set_index('Date', inplace=True)

            # Store the cleaned data
            aligned_data[symbol] = data[['close']]

        except FileNotFoundError:
            logging.error(f"File {symbol}.csv not found.")
            continue
        except Exception as e:
            logging.error(f"An error occurred while processing {symbol}.csv: {e}")
            continue

    # Ensure all data has the same starting date
    if aligned_data:
        max_date = max(df.index.min() for df in aligned_data.values() if not df.empty)
        combined_df = pd.concat([df[df.index >= max_date] for df in aligned_data.values()], axis=1)
        combined_df.columns = [symbol for symbol in aligned_data.keys()]

        # Drop rows with any missing values
        combined_df.dropna(axis=0, how='any', inplace=True)

        # Drop rows where any LTP is zero
        combined_df = combined_df[(combined_df != 0).all(axis=1)]

        # Save the DataFrame to CSV without extra commas and with handling of NaN
        #combined_df.to_csv('check.csv', index=True)  # Saving with index for dates
        #print(combined_df)
        return combined_df
    else:
        logging.error("No valid stock data available.")
        return pd.DataFrame()  # Return an empty DataFrame

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

def is_mean_reverting(spread, confidence_level=0.05):
    """
    Test if a spread is mean-reverting using Augmented Dickey-Fuller test
    and checking the regression coefficient.
    
    Parameters:
        spread: price spread between two assets
        confidence_level: significance level for statistical tests
    
    Returns:
        bool: True if the spread is mean-reverting, False otherwise
    """
    from statsmodels.tsa.stattools import adfuller
    
    try:
        # Perform ADF test
        adf_result = adfuller(spread, autolag='AIC')
        p_value = adf_result[1]
        
        # Check if spread is stationary
        if p_value >= confidence_level:
            logging.info(f"Failed ADF test: p-value {p_value:.4f} >= {confidence_level}")
            return False
        
        # Check regression coefficient
        lagged_spread = spread.shift(1)
        spread_diff = spread - lagged_spread
        spread_diff = spread_diff.dropna()
        lagged_spread = lagged_spread.iloc[1:]
        
        if len(spread_diff) < 2:
            logging.warning("Insufficient data points for mean reversion test")
            return False
            
        model = OLS(spread_diff, lagged_spread).fit()
        coefficient = model.params[0]
        
        # Check if coefficient indicates mean reversion
        if coefficient >= 0:
            logging.info(f"Non-mean-reverting coefficient detected: {coefficient:.4f}")
            return False
            
        # Calculate half-life
        half_life = -np.log(2) / coefficient
        
        # Validate half-life
        if not np.isfinite(half_life) or half_life <= 0:
            logging.info(f"Invalid half-life detected: {half_life}")
            return False
            
        # Check if half-life is within reasonable bounds
        if half_life > config['max_window']:
            logging.info(f"Half-life too long: {half_life:.1f} > {config['max_window']}")
            return False
            
        return True
        
    except Exception as e:
        logging.warning(f"Error in mean reversion test: {str(e)}")
        return False

def calculate_half_life(spread):
    """
    Calculate the half-life of mean reversion for a given spread series.
    Returns a valid integer half-life or config['min_window'] if the spread is not mean-reverting or an error occurs.
    """
        
    try:
        lagged_spread = spread.shift(1)
        spread_diff = spread - lagged_spread
        spread_diff = spread_diff.dropna()
        lagged_spread = lagged_spread.iloc[1:]
        
        model = OLS(spread_diff, lagged_spread).fit()
        coefficient = model.params[0]
        half_life = -np.log(2) / coefficient
        
        # Bound half-life within acceptable range
        half_life = np.clip(half_life, config['min_window'], config['max_window'])
        
        return int(round(half_life))
        
    except Exception as e:
        logging.warning(f"Error in half-life calculation: {str(e)}")
        return config['min_window']  # Fallback to min window value in case of error

def validate_pair(stock_a, stock_b, data):
    try:
        # Calculate the hedge ratio
        hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])
        
        if hedge_ratio is None or not isinstance(hedge_ratio, (int, float)):
            return False, "Invalid hedge ratio", None, None
        
        # Calculate and validate spread
        spread = data[stock_a] - hedge_ratio * data[stock_b]

        if not is_mean_reverting(spread):
            return False, 'Not Mean Reverting', None, None
        
        # Check for NaN or infinite values in the spread
        if spread.isna().any() or np.isinf(spread).any():
            return False, "Invalid spread", None, None
        
        # Calculate the half-life of the spread
        half_life = calculate_half_life(spread)
        
        # Check if half-life is valid (not None and positive)
        if half_life is None or half_life <= 0:
            return False, "Invalid or None half-life", None, None
        
        # Calculate rolling window based on half-life
        rolling_window = int(half_life)
        
        # Ensure rolling window is at least 1
        if rolling_window < 1:
            return False, "Invalid rolling window", half_life, None
        
        # Pair is valid if all checks pass
        return True, "Valid pair", half_life, rolling_window
    
    except Exception as e:
        # Log the error with more context and return False to skip the pair
        logging.error(f"Error validating pair {stock_a}-{stock_b}: {e}")
        return False, f"Error validating pair {stock_a}-{stock_b}: {e}", None, None



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

# Function for Engle-Granger cointegration test
def engle_granger_cointegration(data, p_value_threshold=config['p_value_threshold']):
    stock_symbols = data.columns
    cointegrated_pairs = set()  # Use a set to avoid duplicates
    
    for pair in combinations(stock_symbols, 2):
        sorted_pair = tuple(sorted(pair))  # Ensure pairs are always in the same order
        if sorted_pair not in cointegrated_pairs:  # Only process if not already done
            score, p_value, _ = coint(data[pair[0]], data[pair[1]])
            if p_value < p_value_threshold:
                cointegrated_pairs.add(sorted_pair)  # Add the pair in sorted order
                logging.info(f"Cointegrated Pair: {pair} with p-value: {p_value}")
    
    return list(cointegrated_pairs)

# Function to calculate hedge ratio
def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio
# Store half-lives for stock pairs
half_life_cache = {}

# Function to calculate half-life only once
def get_half_life(stock_a, stock_b, data, cache=half_life_cache):
    """
    Retrieve the cached half-life for stock pair or calculate if not available.
    """
    pair = tuple(sorted([stock_a, stock_b]))  # Ensure the pair is in sorted order for consistency
    if pair in cache:
        # If half-life is already cached, return it
        return cache[pair]
    else:
        # Otherwise, calculate it
        spread = data[stock_a] - calculate_hedge_ratio(data[stock_a], data[stock_b]) * data[stock_b]
        half_life = calculate_half_life(spread)
        cache[pair] = half_life  # Store the calculated half-life
        logging.info(f"Calculated and cached half-life for {stock_a}-{stock_b}: {half_life}")
        return half_life
    
def rolling_mean(arr, window):
    # Compute rolling mean, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].mean() for i in range(len(arr))])

def rolling_std(arr, window):
    # Compute rolling standard deviation, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].std() for i in range(len(arr))])
    
def backtest_stat_arbitrage(data, stock_a, stock_b, initial_capital=10000, 
                          z_entry_threshold_a=-2.0, z_entry_threshold_b=2.0, z_exit_threshold=0.0, 
                          min_holding_period=3, commission_rate=0.001, stop_loss_threshold=0.05, 
                          cooldown_period=1, min_window=config['min_window'], max_window=config['max_window'], half_life=None):
    """
    Backtest the statistical arbitrage strategy using pre-calculated half-life.
    """
    # Use provided half-life if available
    if half_life is None:
        half_life = get_half_life(stock_a, stock_b, data)
    
    # Set rolling window based on the half-life
    rolling_window = int(half_life)

    # Calculate the hedge ratio
    hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])

    # Calculate spread
    spread = data[stock_a] - hedge_ratio * data[stock_b]
    if spread.isna().any():
        logging.warning("NaN values detected in spread calculation")
        spread = spread.fillna(method='ffill')

        # Calculate the hedge ratio
    hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])

    # Calculate spread
    #stock_a_close = data[stock_a].values
    #stock_b_close = data[stock_b].values

    # Compute the spread
    #spread = stock_a_close - hedge_ratio * stock_b_close
    #print(f"Spread: {spread}")

    # Calculate rolling mean and rolling standard deviation
    #spread_mean = rolling_mean(spread, rolling_window)
    #spread_std = rolling_std(spread, rolling_window)
    #z_score = (spread - spread_mean) / spread_std

    # Calculate z-scores
    spread_mean = spread.rolling(window=rolling_window, min_periods=min_window).mean()
    spread_std = spread.rolling(window=rolling_window, min_periods=min_window).std()
    z_score = (spread - spread_mean) / spread_std

    # Replace infinite values and NaN in z-scores
    z_score = z_score.replace([np.inf, -np.inf], np.nan)
    z_score = z_score.fillna(0)


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
            positions_b.iloc[i] = 1  # Go long on stock_b
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
            if (holding_counter_a >= min_holding_period and current_loss_a < -stop_loss_threshold) or \
               (holding_counter_a >= min_holding_period and z_score.iloc[i] >= z_exit_threshold):
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
            if (holding_counter_b >= min_holding_period and current_loss_b < -stop_loss_threshold) or \
               (holding_counter_b >= min_holding_period and z_score.iloc[i] <= z_exit_threshold):
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
            total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio, half_life)




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
        
        # Iterate over different stop-loss thresholds
        for stop_loss_threshold in np.arange(0.05, 0.21, 0.05):
            # Divide data into training and testing sets
            train_start = 0
            train_end = len(data) - walk_forward_window
            test_start = train_end
            test_end = len(data)

            while test_end <= len(data):
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]

                # Backtest on training data
                (cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown,
                 total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio, half_life) = backtest_stat_arbitrage(
                    train_data,
                    stock_a,
                    stock_b,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    min_window=min_window,
                    max_window=max_window
                )


                # Evaluate on testing data
                (test_cumulative_returns, _, _, _, _, _, _, _, _, _, _) = backtest_stat_arbitrage(
                    test_data,
                    stock_a,
                    stock_b,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    min_window=min_window,
                    max_window=max_window
                )

                risk_adjusted_return = sharpe_ratio / (1 + max_drawdown) if max_drawdown != 0 else 0

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
                    'hedge_ratio': hedge_ratio,
                    'half_life': half_life,  # Add this line to store the rolling window
                })

                # Update training and testing windows
                train_start += walk_forward_window
                train_end += walk_forward_window
                test_start += walk_forward_window
                test_end += walk_forward_window

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
                cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown, total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio, half_life = backtest_stat_arbitrage(
                    data=process_stock_data([stock_a, stock_b], fill_method='ffill'),  # Process data for this pair
                    stock_a=stock_a,
                    stock_b=stock_b,
                    z_entry_threshold_a=best_result['z_entry_threshold_a'],
                    z_entry_threshold_b=best_result['z_entry_threshold_b'],
                    z_exit_threshold=best_result['z_exit_threshold'],
                    stop_loss_threshold=best_result['stop_loss_threshold']
                )

                all_final_results.append({
                    'Stock A': stock_a,
                    'Stock B': stock_b,
                    'Z Entry Threshold A': best_result['z_entry_threshold_a'],
                    'Z Entry Threshold B': best_result['z_entry_threshold_b'],
                    'Z Exit Threshold': best_result['z_exit_threshold'],
                    'Stop Loss Threshold': best_result['stop_loss_threshold'],
                    'Final Cumulative Returns': cumulative_returns.iloc[-1],
                    'Final Sharpe Ratio': sharpe_ratio,
                    'Final Sortino Ratio': sortino_ratio,
                    'Final Max Drawdown': max_drawdown,
                    'Cumulative Returns Series': cumulative_returns.values.tolist(),
                    'Total Trades': total_trades,
                    'Win Rate': win_rate,
                    'Profit Factor': profit_factor,
                    'Average Profit': average_profit,
                    'Average Loss': average_loss,
                    'Hedge Ratio': hedge_ratio,
                    'Half Life': half_life
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
