import pandas as pd
import logging
import os
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations
from statsmodels.regression.linear_model import OLS
from typing import List, Optional


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to get all filenames from the 'datas' folder
def get_stock_filenames(data_folder='/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/'):
    filenames = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    stock_symbols = [f.replace('.csv', '') for f in filenames]
    return stock_symbols

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
    #output_filename = f'{stock_symbols[0]}_{stock_symbols[1]}_combined_data.csv'
    #combined_df.to_csv(output_filename)
    
    logging.info(f"Processed data for {len(stock_symbols)} stocks")
    return combined_df

# Calculate hedge ratio
def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio

# Function to perform ADF test with error handling
def perform_adf_test(series, significance_level=0.05):
    try:
        # Perform ADF test with reduced lags for shorter series
        result = adfuller(series, maxlag=1)  # Reduce maxlag for shorter series
        adf_stat, p_value = result[0], result[1]
        return p_value < significance_level, p_value
    except ValueError as e:
        logging.warning(f"ADF Test failed: {str(e)}")
        return False, 1.0  # Return non-stationary result if test fails
    except Exception as e:
        logging.error(f"Unexpected error in ADF test: {str(e)}")
        return False, 1.0

# Function for Engle-Granger cointegration test with collinearity check and ADF test
def engle_granger_cointegration(pair_data, p_value_threshold=0.05, correlation_threshold=0.95, min_periods=252):  # 252 trading days = 1 year
    # Check if we have enough data
    if len(pair_data) < min_periods:
        logging.warning(f"Insufficient data points ({len(pair_data)}). Minimum required: {min_periods}")
        return None

    stock1, stock2 = pair_data.columns
    # Check for high correlation, which can indicate collinearity
    correlation = pair_data[stock1].corr(pair_data[stock2])
    if abs(correlation) > correlation_threshold:
        logging.warning(f"Skipping pair ({stock1}, {stock2}) due to high correlation ({correlation})")
        return None

    # Perform cointegration test
    try:
        score, p_value, _ = coint(pair_data[stock1], pair_data[stock2])
    except Exception as e:
        logging.error(f"Cointegration test failed for ({stock1}, {stock2}): {str(e)}")
        return None

    hedge_ratio = calculate_hedge_ratio(pair_data[stock1], pair_data[stock2])
    spread = pair_data[stock1] - hedge_ratio * pair_data[stock2]
    
    # Perform ADF test on the spread
    is_stationary, adf_p_value = perform_adf_test(spread)
    if not is_stationary:
        logging.info(f"Pair ({stock1}, {stock2}) spread is not stationary. ADF p-value: {adf_p_value}")
        return None

    # Check for mean reversion
    lagged_spread = spread.shift(1)
    spread_diff = spread - lagged_spread
    spread_diff = spread_diff.dropna()
    lagged_spread = lagged_spread.iloc[1:]
    model = OLS(spread_diff, lagged_spread).fit()
    coefficient = model.params[0]
    
    # Exclude non-mean-reverting pairs (p-value >= threshold)
    if p_value < p_value_threshold and coefficient < 0:
        logging.info(f"Cointegrated Pair: ({stock1}, {stock2})")
        logging.info(f"Cointegration p-value: {p_value}")
        logging.info(f"ADF p-value: {adf_p_value}")
        logging.info(f"Mean reversion coefficient: {coefficient}")
        logging.info(f"Number of observations: {len(pair_data)}")
        return (stock1, stock2, p_value, adf_p_value, coefficient, len(pair_data))
    else:
        logging.info(f"Non-cointegrated pair ({stock1}, {stock2})")
        logging.info(f"Cointegration p-value: {p_value}")
        logging.info(f"Mean reversion coefficient: {coefficient}")
        return None

def main():
    # Get all stock filenames from the datas folder
    stock_symbols = get_stock_filenames()
    results = []  # To store the results

    # Iterate over each unique pair of stocks
    for stock1, stock2 in combinations(stock_symbols, 2):
        # Process and align data for both stocks
        combined_data = process_stock_data([stock1, stock2])
        print(combined_data)

        # If combined data is valid, run cointegration test
        if not combined_data.empty:
            result = engle_granger_cointegration(combined_data)
            if result:
                results.append(result)

    # Save the results to a CSV
    if results:
        results_df = pd.DataFrame(
            results, 
            columns=['stock_a', 'stock_b', 'Cointegration_P-Value', 'ADF_P-Value', 
                    'Mean_Reversion_Coefficient', 'Observations']
        )
        results_df.to_csv('cointegration_results.csv', index=False)
        logging.info("Cointegration test completed. Results saved to 'cointegration_results.csv'.")
    else:
        logging.info("No cointegrated pairs found.")


if __name__ == "__main__":
    main()