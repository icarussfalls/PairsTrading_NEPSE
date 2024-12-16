import requests
import json
import pandas as pd
import os

# Define stock symbol groups
stocks_banking = ['NABIL', 'KBL', 'MBL', 'SANIMA', 'NICA']
stocks_finance = ['CFCL', 'GFCL', 'MFIL', 'GUFL', 'NFL']
stocks_microfinance = ['CBBL', 'DDBL', 'SKBBL', 'SMFDB', 'SWBBL', 'SMB', 'FOWAD', 'KLBSL']
stocks_life = ['ALICL', 'LICN', 'NLIC', 'CLI', 'ILI', 'SJLIC']
stocks_non_life = ['NICL', 'NIL', 'NLG', 'SICL', 'PRIN', 'HEI']
stocks_others = ['NRIC', 'NTC', 'SHIVM', 'HRL']
stocks_hydro = ['NYADI', 'RADHI', 'NHPC', 'KPCL', 'HDHPC', 'DHPL', 'API', 'AKPL', 'UNHPL']

# Concatenate all stock symbols into one list
stock_symbols = stocks_banking + stocks_finance + stocks_microfinance + stocks_life + stocks_non_life + stocks_hydro + stocks_others

# Base URL for data request
url = "https://nepsealpha.com/trading/1/history"

# Set up request headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://nepsealpha.com/trading/chart?symbol=',
    'X-Requested-With': 'XMLHttpRequest',
    'DNT': '1'
}

# Define cookies from the provided headers
cookies = {
    '_ga_9XQG87E8PF': 'GS1.2.1725901990.1.0.1725901990.0.0.0',
    'remember_web_59ba36addc2b2f9401580f014c7f58ea4e30989d': 'eyJpdiI6IjJYOU14Z2Z6OFRWSzF4Unh4ZDhodEE9PSIsInZhbHVlIjoiMGxBODBkaVJSN3dtTVVVNlBnRVpRTUQ0RnJKVHowT0R1eWx6UUdzc05pYlNoaitpclBtNGpqb0phRWdiQUt5NXUwdCtJa2xzMWgxcnpQdXQ1ZnhSbEE1RS9iNjliNnlhYmN2K0FMbmhneWJVYmtXY2FOQnl6ZUVHV0k5UHEyeXg5R2ROeGFtRS9zNFdFRFJoalJscmx1N1lYSGdwWkpxNG0xNWxER0x2dG95RlNVbDJNVXJrY2pzTWdSaFRHRmt3c0tvTjZFR2pzMEZVUVFUd3ZJdG5NSGxjWitTRnExL1RReHlQQUlRMU50Zz0iLCJtYWMiOiI5NTVjODgyN2QwZjIyMDEzODFkNjU3OTcxOTI0ZWZiYmNkYmU3M2Q3MGQ2Y2U3MzFjMjUzODAzMmUyOThlMzRlIiwidGFnIjoiIn0%3D',
    '_ga': 'GA1.1.1394048420.1722281423',
    'site_theme': 'light',
    '__cf_bm': 'RYFuyaXK83mcyprcqWzHahSXzupJoEKm0hCGV6vriuc-1730991287-1.0.1.1-XzaYTtGqDwWvkDkuK6WZL_wmWQLE7jS0lYh8J3zSpBsnjKEtQ_.OmxpDqPPlJpkGZog2SXR5Ff_B.3Zz8fERow',
    'cf_clearance': 'Pp0xJI.jxiYx1x0xHfimZWv_QCJVq958a4THO4RMpIE-1730991295-1.2.1.1-auEKNaMhqJMy2hozYb_0XYWze6T5DCOzMVBlH3s9o0BytF5U1mk3aRsDgTyW5dtHVKKMXbh3fbVYzWMFpecyw64Uu_t5Yg04Js4EQ08muVEQXIWkinYJnO_b7OigjtXz9KehrRozfttIkuA6sXop5YgqvaxENFQvNPCyEeWIdZHtGMjcFu7xOSA9y7eL8R.FteJZ9bxoi.nqIKEWX7BG55gV9Zap3GSycaWUl9yU5TJOz_17n6n.OBoSXYWSXv.u1H20Yy7k0p0AgbxlvAiKhpC4.07YUjEMHRRG1RTh370X93f4N32dwEC7MQkRmuNo8bIM9YrmEudUJl3IIQVL0ROzhfcYe.UrRAjArg__XSlnOeQyyRPQ2srsw2J0AjXIebY79LRc6OvexG3jzMvJHA',
    'nepsealpha_session': 'eyJpdiI6Ikk3bzYwS2tma29LZUJOSnBTYlduWUE9PSIsInZhbHVlIjoic3IybU5RdnZha3d5RXhvUkJrRkFjcllqWlNSNDkrYkZoZldvM0R4Mm9RbXBweDVGUFNvdVozSE5KNElPVFExUi9yN01aNjl6YnBzMk1idFl5TnNxQkVEeHg0WGtzbUtkUzFZNTUyczhxWE9NWWxieS9jU3U4R1NFREFscU5ncnUiLCJtYWMiOiJlNTZjZDM1ODdkZjA4MGIwNWYyNTc1OGZmOThkMGRhYWI4YTU1YzQyZGNjZjQ3MDQyZTVjNTU1ZmQ4MWZmMjRhIiwidGFnIjoiIn0%3D',
}

# Start a session to maintain cookies across requests
session = requests.Session()

# Loop through each stock symbol and retrieve data
for symbol in stock_symbols:
    payload = {
        'fsk': 'fJYyOyExM7wifJph',
        'symbol': symbol,
        'resolution': '1D',
        'pass': 'ok'
    }

    try:
        # Send a GET request
        response = session.get(url, params=payload, headers=headers, cookies=cookies)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()  # Parse JSON directly
            df = pd.DataFrame(data)  # Create DataFrame from data

            # Rename and process columns
            df.columns = ['Status', 'Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']
            df = df.drop(columns=['Status'])  # Drop the 'Status' column

            # Convert OHLCV columns to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert Timestamp to datetime and sort data
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
            df = df.drop(columns=['Timestamp']).sort_values(by='Date')

            # Define filename and save DataFrame as CSV
            filename = f"{symbol}.csv"
            path = os.path.join('datas', filename)
            os.makedirs('datas', exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Data saved for {symbol} in {path}")
        else:
            print(f"Failed to retrieve data for {symbol}: {response.status_code}")
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
