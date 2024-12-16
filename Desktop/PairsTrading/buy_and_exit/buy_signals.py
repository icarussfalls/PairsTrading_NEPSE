import pandas as pd
import numpy as np
import glob
import os

def rolling_mean(arr, window):
    # Compute rolling mean, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].mean() for i in range(len(arr))])

def rolling_std(arr, window):
    # Compute rolling standard deviation, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].std() for i in range(len(arr))])


def get_latest_signal(stock_a, stock_b, half_life, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name, hedge_ratio):
    rolling_window = int(half_life)

    # Calculate spread
    stock_a_close = stock_a['close'].values
    stock_b_close = stock_b['close'].values

    # Compute the spread
    spread = stock_a_close - hedge_ratio * stock_b_close

    # Calculate rolling mean and rolling standard deviation
    spread_mean = rolling_mean(spread, rolling_window)
    spread_std = rolling_std(spread, rolling_window)

    # Debug: check the rolling window calculations
    #print(f"Spread mean: {spread_mean}")
    #print(f"Spread std: {spread_std}")

    # Compute Z-score, avoiding division by zero
    # Remove NaN values for mean and std (rolling window handling)
    valid_indices = ~np.isnan(spread_mean) & ~np.isnan(spread_std)  # indices where both mean and std are not NaN
    z_score = (spread[valid_indices] - spread_mean[valid_indices]) / spread_std[valid_indices]

    # Debug: Check the length and last few Z-scores
    print(f"Z-score Length: {len(z_score)}")
    print(z_score[-5:])  # Debug: Check the last 5 Z-scores

    # Check for sufficient data to generate a signal
    if len(z_score) < 2:
        print("Insufficient data for signal generation")
        return 'Hold', 'None', np.nan

    # Use the last two values of z_score
    latest_z_score = z_score[-1]
    previous_z_score = z_score[-2]
    print(f"Latest Z-score: {latest_z_score:.2f}, Previous Z-score: {previous_z_score:.2f}")

    # Generate signal based on Z-score thresholds
    if latest_z_score < z_entry_thresh_a and previous_z_score > z_entry_thresh_a and latest_z_score < z_exit_thresh :
        return 'Buy', stock_a_name, stock_b_name, latest_z_score
    elif latest_z_score > z_entry_thresh_b and previous_z_score < z_entry_thresh_b and latest_z_score > z_exit_thresh:
        return 'Buy', stock_b_name, stock_a_name, latest_z_score
    elif previous_z_score < z_exit_thresh <= latest_z_score:
        return 'Exit', stock_a_name, stock_b_name, latest_z_score
    elif previous_z_score > -z_exit_thresh >= latest_z_score:
        return 'Exit', stock_b_name, stock_a_name, latest_z_score
    else:
        return 'Hold', 'None', 'None', latest_z_score



def main():
    # Load summary table
    summary_table = pd.read_csv('Pairs Trading/signals/results_summary_filtered.csv')
    
    # Lists and dictionaries for storing results
    all_signals = []
    buy_count = {}
    total_sharpe = {}
    total_win_rate = {}
    
    lookback_period = 30  # Adjust this value as needed

    # Process each stock pair
    for index, row in summary_table.iterrows():
        stock_a_name = row['Stock A']
        stock_b_name = row['Stock B']
        
        try:
            # Load stock data
            stock_a_data = pd.read_csv(f'datas/{stock_a_name}.csv')
            stock_b_data = pd.read_csv(f'datas/{stock_b_name}.csv')
            
            # Take lookback period from each stock independently
            stock_a_data = stock_a_data.tail(lookback_period).copy()
            stock_b_data = stock_b_data.tail(lookback_period).copy()

            if len(stock_a_data) < lookback_period or len(stock_b_data) < lookback_period:
                print(f"Error: Insufficient data for {stock_a_name} or {stock_b_name}")
                continue  # Skip to the next pair
            
            # Get signal
            signal, stock_to_buy, other_pair, latest_z_score = get_latest_signal(
                stock_a_data,
                stock_b_data,
                row['Half Life'],
                row['Z Entry Threshold A'],
                row['Z Entry Threshold B'],
                row['Z Exit Threshold'],
                stock_a_name,
                stock_b_name,
                row['Hedge Ratio']
            )
            
            # Store results only if we got a valid signal
            if latest_z_score is not None:
                all_signals.append({
                    'Stock A': stock_a_name,
                    'Stock B': stock_b_name,
                    'Signal': signal,
                    'Stock': stock_to_buy,
                    'Latest Z-Score': latest_z_score,
                    'Rolling Window': row['Half Life'],
                    'Z Entry Threshold A': row['Z Entry Threshold A'],
                    'Z Entry Threshold B': row['Z Entry Threshold B'],
                    'Z Exit Threshold': row['Z Exit Threshold']
                })
                
                if signal == 'Buy':
                    if stock_to_buy not in buy_count:
                        buy_count[stock_to_buy] = 0
                        total_sharpe[stock_to_buy] = 0
                        total_win_rate[stock_to_buy] = 0
                    buy_count[stock_to_buy] += 1
                    total_sharpe[stock_to_buy] += row['Final Sharpe Ratio']
                    total_win_rate[stock_to_buy] += row['Win Rate']
                    
        except Exception as e:
            print(f"Error processing pair {stock_a_name}-{stock_b_name}: {str(e)}")
            continue

    # Save `all_signals` to CSV
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_output_path = os.path.join(os.getcwd(), 'Pairs Trading/signals/all_signals.csv')
        signals_df.to_csv(signals_output_path, index=False)
        print(f"All signals saved to: {signals_output_path}")

    # Create ranking DataFrame if we have any buy signals
    if buy_count:
        ranking_df = pd.DataFrame({
            'Stock': list(buy_count.keys()),
            'Buy Count': list(buy_count.values()),
            'Average Sharpe Ratio': [total_sharpe[stock] / buy_count[stock] for stock in buy_count],
            'Win Rate': [total_win_rate[stock] / buy_count[stock] for stock in buy_count]
        })
        
        ranking_df['Half Life'] = ranking_df['Stock'].apply(lambda x: 
            summary_table.loc[summary_table['Stock A'] == x, 'Half Life'].values[0] 
            if x in summary_table['Stock A'].values 
            else summary_table.loc[summary_table['Stock B'] == x, 'Half Life'].values[0]
        )
        
        ranking_df.sort_values(by=['Half Life', 'Average Sharpe Ratio', 'Win Rate'], 
                             ascending=[True, False, False], 
                             inplace=True)
        
        print("\nRanked Stocks to Buy by Half Life:")
        print(ranking_df)
        
        ranking_output_path = os.path.join(os.getcwd(), 'Pairs Trading/signals/ranked_stocks_to_buy_by_half_life.csv')
        ranking_df.to_csv(ranking_output_path, index=False)
        print(f"Ranking saved to: {ranking_output_path}")
    else:
        print("No buy signals generated")


if __name__ == "__main__":
    main()