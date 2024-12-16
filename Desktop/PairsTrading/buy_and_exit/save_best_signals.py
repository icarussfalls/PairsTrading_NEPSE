import pandas as pd
import os
import ast

# Load the file
file_path = '/Users/icarus/Desktop/QuantatitiveTradingStrategies/Pairs Trading/results/cointegrated_pairs_results.csv'
try:
    all_results_df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")

# Required columns
required_columns = [
    'Stock A', 'Stock B', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]

# Check if required columns are present
missing_columns = [col for col in required_columns if col not in all_results_df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in input file: {missing_columns}")

# Extract relevant data
summary_table = all_results_df[required_columns]

# Function to extract numerical values from strings
def extract_first_value(value):
    try:
        value = str(value).strip()
        if value.startswith("(") and value.endswith(")"):
            return float(ast.literal_eval(value)[0])  # Convert tuple-like string
        return float(value)  # Handle normal numeric values
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Warning: Failed to process value {value}: {e}")
        return None

# Apply to specified columns
columns_to_convert = [
    'Final Cumulative Returns', 'Final Sharpe Ratio', 'Final Sortino Ratio', 
    'Final Max Drawdown', 'Total Trades', 'Profit Factor', 
    'Average Profit', 'Average Loss'
]
for col in columns_to_convert:
    summary_table[col] = summary_table[col].apply(extract_first_value)

# Filter rows based on conditions
summary_table = summary_table[summary_table['Final Sharpe Ratio'] > 1.5]
summary_table = summary_table[
    (summary_table['Average Profit'] > summary_table['Average Loss']) &
    (summary_table['Profit Factor'] > 2) &
    (summary_table['Win Rate'] > 0.70) &
    (summary_table['Final Cumulative Returns'] > 1)
]

# Round numeric columns
numeric_columns = [
    'Z Entry Threshold A', 'Z Entry Threshold B', 'Z Exit Threshold',
    'Stop Loss Threshold', 'Final Cumulative Returns', 'Final Sharpe Ratio',
    'Final Sortino Ratio', 'Final Max Drawdown', 'Total Trades', 
    'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]
summary_table[numeric_columns] = summary_table[numeric_columns].round(2)

# Remove rows with inf or 0 in numeric columns
summary_table = summary_table[~summary_table[numeric_columns].isin([float('inf'), 0]).any(axis=1)]

# Save filtered data
output_path = os.path.join(os.getcwd(), 'Pairs Trading/signals/results_summary_filtered.csv')
summary_table.to_csv(output_path, index=False)
print(f"Filtered results saved to {output_path}")
