import requests
import json
import time
import pandas as pd
from datetime import datetime
import os

# Read the company list from a file
with open("company_list.json", "r") as file:
    companies = json.load(file)

# Define the base URL for fetching the chart data
base_url = "https://merolagani.com/handlers/TechnicalChartHandler.ashx?type=get_advanced_chart"

# Function to convert date to Unix timestamp
def to_unix_timestamp(date_str):
    """Convert a date string (YYYY-MM-DD) to a Unix timestamp."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return int(date_obj.timestamp())
    except Exception as e:
        print(f"Error converting date {date_str} to timestamp: {e}")
        return None

# Function to check if the most recent data is from the current month
def is_data_from_current_month(data_timestamps):
    """Check if the latest timestamp is from the current month."""
    if not data_timestamps:
        return False
    
    # Convert the latest timestamp to a datetime object
    latest_timestamp = data_timestamps[-1]
    latest_date = datetime.utcfromtimestamp(latest_timestamp)
    
    # Compare the latest date's year and month with the current year and month
    current_date = datetime.now()
    return latest_date.year == current_date.year and latest_date.month == current_date.month

# Iterate over all companies in the list
for company in companies:
    symbol = company["d"]  # Company symbol (e.g., ADBL, CBL)
    
    # Skip companies whose symbol ends with 'PO' (promoter shares)
    if symbol.endswith("PO"):
        print(f"Skipping {company['l']} ({symbol}) because it ends with 'PO'.")
        continue
    
    print(f"Fetching data for {company['l']} ({symbol})...")
    
    earliest_date = "2000-01-01"  # Lets start with the earliest date to cover all data for individual stocks
    range_start_date = to_unix_timestamp(earliest_date)  # Convert to Unix timestamp
    range_end_date = int(time.time())  # Use current timestamp for rangeEndDate
    
    if range_start_date is None:
        print(f"Skipping {symbol} due to invalid start date.")
        continue
    
    # Construct the full URL with the dynamic rangeStartDate and rangeEndDate
    url = f"{base_url}&symbol={symbol}&resolution=1D&rangeStartDate={range_start_date}&rangeEndDate={range_end_date}&from=&isAdjust=1&currencyCode=NPR"
    
    try:
        # Send the GET request for the company's technical chart data
        response = requests.get(url)
        
        # Check if the response was successful
        if response.status_code == 200:
            data = response.json()

            # Extract the relevant data (open, high, low, close, volume, and timestamp)
            if 'c' in data and 'h' in data and 'l' in data and 'o' in data and 't' in data and 'v' in data:
                chart_data = {
                    'timestamp': data['t'],
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                }

                # Check if the latest data is from the current month
                if not is_data_from_current_month(chart_data['timestamp']):
                    print(f"Skipping {symbol} because there is no data for the current month.")
                    continue

                # Create a DataFrame from the chart data
                df = pd.DataFrame(chart_data)

                # Convert the timestamp to a readable date format
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                df.drop(columns=['timestamp'], inplace=True)  # Drop the raw timestamp column

                # Save the data to a CSV file specific to the company
                # Define filename and save DataFrame as CSV
                filename = f"{symbol}.csv"
                path = os.path.join('datas', filename)
                os.makedirs('datas', exist_ok=True)
                df.to_csv(path, index=False)
                print(f"Data saved for {symbol} in {path}")
        else:
            print(f"Failed to retrieve data for {symbol}, Status Code: {response.status_code}")
    
    except Exception as e:
        print(f"Error occurred while fetching data for {symbol}: {e}")
    
    # To avoid hitting request limits, wait for 1 second before making the next request
    #time.sleep(1)

print("Process completed!")
