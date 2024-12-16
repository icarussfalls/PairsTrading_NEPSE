import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize

def setup_logging():
    """
    Configure logging with a more informative format.
    """
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_stock_data(
    stock_symbols: List[str], 
    data_directory: str = '/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/', 
    fill_method: Optional[str] = 'ffill',
    lookback_period: int = 252
) -> pd.DataFrame:
    """
    Enhanced stock data processing with more robust error handling and flexible lookback.
    
    Args:
        stock_symbols (List[str]): List of stock symbols to process
        data_directory (str): Path to directory containing stock CSV files
        fill_method (Optional[str]): Method to fill missing values
        lookback_period (int): Number of most recent trading days to retain
    
    Returns:
        pd.DataFrame: Aligned stock closing prices
    """
    setup_logging()
    
    if len(stock_symbols) < 2:
        logging.error("At least two stock symbols required for pair trading")
        return pd.DataFrame()

    stock_dataframes = {}
    
    for symbol in stock_symbols:
        try:
            # Enhanced file reading with more robust date handling
            df = pd.read_csv(
                f'{data_directory}{symbol}.csv', 
                parse_dates=['date'], 
                index_col='date'
            )
            
            # Comprehensive data cleaning
            df.index = pd.to_datetime(df.index).date
            df = df.loc[~df.index.duplicated(keep='first')]
            df = df[(df['close'] != 0) & (df['close'].notna())]
            
            # Sort and retain most recent data
            df = df.sort_index()[-lookback_period:]
            
            stock_dataframes[symbol] = df[['close']]
        
        except FileNotFoundError:
            logging.error(f"File not found for {symbol}")
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if len(stock_dataframes) < 2:
        logging.error("Insufficient valid stock data for pair trading")
        return pd.DataFrame()

    # Find common date range with precise alignment
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
    
    logging.info(f"Processed data for {len(stock_symbols)} stocks")
    return combined_df

def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    """
    Calculate hedge ratio using Ordinary Least Squares regression.
    
    Args:
        stock_a_prices (pd.Series): Prices of stock A
        stock_b_prices (pd.Series): Prices of stock B
    
    Returns:
        float: Calculated hedge ratio
    """
    try:
        model = OLS(stock_a_prices, stock_b_prices).fit()
        return model.params[0]
    except Exception as e:
        logging.error(f"Hedge ratio calculation error: {e}")
        return 1.0  # Default hedge ratio if calculation fails

def ou_process_params(spread):
    """
    Fit an Ornstein-Uhlenbeck process to the spread.
    
    Args:
        spread (pd.Series): Price spread between stocks
    
    Returns:
        Tuple of OU process parameters (theta, mu, sigma)
    """
    def ou_objective(params):
        theta, mu, sigma = params
        dt = 1  # Daily time step
        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        
        predicted_diff = theta * (mu - spread_lag) * dt
        residual = spread_diff - predicted_diff
        
        return np.sum(residual**2)
    
    initial_guess = [0.1, spread.mean(), spread.std()]
    
    try:
        result = minimize(ou_objective, initial_guess, bounds=[(0, None), (None, None), (0, None)])
        theta, mu, sigma = result.x
        return theta, mu, sigma
    except Exception as e:
        logging.error(f"OU process parameter estimation failed: {e}")
        return 0.1, spread.mean(), spread.std()

def calculate_half_life(theta):
    """
    Calculate half-life from mean-reversion rate.
    
    Args:
        theta (float): Mean-reversion rate
    
    Returns:
        float: Half-life of the mean-reversion process
    """
    return np.log(2) / theta if theta > 0 else float('inf')

def generate_trading_signal(
    merged_data: pd.DataFrame, 
    stock_a_name: str, 
    stock_b_name: str, 
    z_entry_a: float, 
    z_entry_b: float, 
    z_exit: float
) -> Dict[str, Any]:
    """
    Generate comprehensive trading signal for a stock pair.
    
    Args:
        merged_data (pd.DataFrame): Merged stock price data
        stock_a_name (str): Name of first stock
        stock_b_name (str): Name of second stock
        z_entry_a (float): Entry threshold for stock A
        z_entry_b (float): Entry threshold for stock B
        z_exit (float): Exit threshold
    
    Returns:
        Dict with signal details
    """
    # Calculate hedge ratio
    hedge_ratio = calculate_hedge_ratio(
        merged_data[stock_a_name], 
        merged_data[stock_b_name]
    )
    
    # Calculate spread
    spread = merged_data[stock_a_name] - hedge_ratio * merged_data[stock_b_name]
    
    # Estimate OU process parameters
    theta, mu, _ = ou_process_params(spread)
    half_life = calculate_half_life(theta)
    rolling_window = max(int(half_life), 5)
    
    # Z-score calculation
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    z_score = (spread - spread_mean) / spread_std
    
    # Retrieve latest and previous Z-scores
    if len(z_score) < 2:
        return {
            'signal': 'Hold', 
            'stock_to_trade': None, 
            'z_score': np.nan,
            'half_life': half_life
        }
    
    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2]
    
    # Signal generation logic
    if latest_z_score <= z_entry_a and previous_z_score > z_entry_a:
        return {
            'signal': 'Buy', 
            'stock_to_trade': stock_a_name, 
            'z_score': latest_z_score,
            'half_life': half_life
        }
    elif latest_z_score >= z_entry_b and previous_z_score < z_entry_b:
        return {
            'signal': 'Buy', 
            'stock_to_trade': stock_b_name, 
            'z_score': latest_z_score,
            'half_life': half_life
        }
    elif (previous_z_score < z_exit <= latest_z_score):
        return {
            'signal': 'Exit', 
            'stock_to_trade': stock_a_name, 
            'z_score': latest_z_score,
            'half_life': half_life
        }
    elif (previous_z_score > -z_exit >= latest_z_score):
        return {
            'signal': 'Exit', 
            'stock_to_trade': stock_b_name, 
            'z_score': latest_z_score,
            'half_life': half_life
        }
    
    return {
        'signal': 'Hold', 
        'stock_to_trade': None, 
        'z_score': latest_z_score,
        'half_life': half_life
    }

def run_pairs_trading_strategy(
    summary_table_path: str = 'Pairs Trading/signals/results_summary_filtered.csv',
    output_path: str = 'pairs_trading_signals.csv',
    min_trade_interval: int = 10
) -> None:
    """
    Execute comprehensive pairs trading strategy with full signal tracking.
    
    Args:
        summary_table_path (str): Path to pairs summary table
        output_path (str): Path to save trading signals
        min_trade_interval (int): Minimum trading interval between signals
    """
    setup_logging()
    
    # Load strategy configuration
    summary_table = pd.read_csv(summary_table_path)
    
    # Initialize tracking variables
    all_signals = []
    open_positions = {}
    trade_count = {}
    
    # Process each stock pair
    for _, row in summary_table.iterrows():
        stock_a, stock_b = row['Stock A'], row['Stock B']
        
        # Preprocess stock data
        processed_data = process_stock_data([stock_a, stock_b])
        if processed_data.empty:
            logging.warning(f"Skipping pair {stock_a}-{stock_b}")
            continue
        
        # Extract trading parameters
        z_entry_a = row['Z Entry Threshold A']
        z_entry_b = row['Z Entry Threshold B']
        z_exit = row['Z Exit Threshold']
        
        # Generate trading signal
        signal = generate_trading_signal(
            processed_data, 
            stock_a, 
            stock_b, 
            z_entry_a, 
            z_entry_b, 
            z_exit
        )
        
        # Prepare signal entry
        signal_entry = {
            'Timestamp': processed_data.index[-1],
            'Stock A': stock_a,
            'Stock B': stock_b,
            'Signal': signal['signal'],
            'Buy Stock': signal['stock_to_trade'] if signal['signal'] == 'Buy' else None,
            'Exit Stock': signal['stock_to_trade'] if signal['signal'] == 'Exit' else None,
            'Z-Score': signal['z_score'],
            'Half-Life': signal['half_life']
        }
        
        # Always add the signal, regardless of type
        all_signals.append(signal_entry)
        
        # Manage open positions for Buy and Exit signals
        pair_key = tuple(sorted([stock_a, stock_b]))
        
        if signal['signal'] == 'Buy':
            # Manage trade frequency and avoid repeated trades
            if ((pair_key not in trade_count) or 
                (trade_count.get(pair_key, 0) < min_trade_interval)):
                open_positions[pair_key] = signal
                trade_count[pair_key] = trade_count.get(pair_key, 0) + 1
        
        elif signal['signal'] == 'Exit':
            if pair_key in open_positions:
                del open_positions[pair_key]
    
    # Save signals to CSV
    signals_df = pd.DataFrame(all_signals)
    signals_df.to_csv(output_path, index=False)
    logging.info(f"Generated {len(signals_df)} total signals")
    logging.info(f"Signals saved to {output_path}")

if __name__ == "__main__":
    run_pairs_trading_strategy()