import requests
import pandas as pd
from datetime import datetime
import os

# Function to retrieve stock data
def get_stock_data(symbol, start_date='2000-10-08'):
    # Get today's date in the format YYYY-MM-DD
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Define the URL for the POST request
    url = "https://nepsealpha.com/nepse-data"

    # Define the payload
    payload = {
        'symbol': symbol,
        'specific_date': end_date,
        'start_date': start_date,
        'end_date': end_date,
        'filter_type': 'date-range',
        'time_frame': 'daily',
        '_token': 'MR4WWhzNlDxVLZxnINlg8bHeQvssjC7V4FkJ2n2C',  # Replace with actual token if necessary
    }

    # Define the headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://nepsealpha.com',
        'Referer': 'https://nepsealpha.com/nepse-data',
        'X-Requested-With': 'XMLHttpRequest',
    }

    # Define the cookies from your browser session
    cookies = {
        '_ga': 'GA1.1.1394048420.1722281423',  # Replace with actual cookie
        '_gid': 'GA1.2.1199659646.1728364222',  # Replace with actual cookie
        'nepsealpha_session': 'eyJpdiI6IkpMbFFMdEVoUEpFWDIreVJuWGJha3c9PSIsInZhbHVlIjoiU1VBRDNwczhWNXpMWnE2SE9MbHZ5VEsvNENINDNwN1JRaE82aDNlL0ZURGl3UEJSQUV3OEp2RzVxYXVzS1pvamxwcGxEQ3BSSUtienIxeTNGNGs4TFI5WnAyNkY5TGl4eGFEc3pHZDBkQmZxbXpQVXM3Z0JVUDBqdDBzQ1pBc0YiLCJtYWMiOiJkZDQ1N2NiMTliZTM4YmRjZjkzYzIxMTM0ZmUwZGZjODJkYTQ0ZDgxMDdlMDUyMDViMDg4MWQ2YjAyMGQ0ZjQ3IiwidGFnIjoiIn0%3D',  # Replace with actual session cookie
        'cf_clearance': '1tCWlcnUEkgA2HanLf_qJtuqCOzoFO0dfbcsHRJmkds-1728365076-1.2.1.1-a8ANa5tDdW7C9ncCODMvbfIvI9rR1YfBzPdbvhF8yjcs0vwaJnJS8BqRPM_x4pmOkxgJplF0PDyvjwsol_VREs7uzsuI_n6IA3BwZu0uIP5IKry0lO4TDGskjxQPky8.4KMaUOFdXuQsgbKId4R_Rdy2fI1ImGYG499KWTgxQEVVVh__MhJC0CEnJhyNjfu66I_v8u1bhU8Po2tgqhpxIjQEdyJDKLioVaUoCPn5jd1z9GF1Ai4IPET4EnMvx2Ed2myA.LOZrCeHboQlGkyE0NQThfHTYsQAh2jcBkkFkyuAiqDTdRTkMfqfcsDOjKhloQsVF_cPJxuviaC8DeT8uqWDbNLA6ScDY3vKzmic_iL.PTP5uHkqz1UD5tGvxnoAVujAY6aP.9eGwTYtfPuXSA',  # Replace with actual cookie
    }

    # Send the POST request
    response = requests.post(url, headers=headers, cookies=cookies, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Debug print the response data
        #print(f"Response Data for {symbol}:", data)

        # Check if data is a list or a dictionary containing the list
        if isinstance(data, dict) and 'data' in data:
            data = data['data']  # Adjust this line if the relevant data is nested in the response

        # Create a DataFrame from the response data
        df = pd.DataFrame(data)

        # Debug print the DataFrame columns
        print(f"DataFrame Columns for {symbol}:", df.columns)

        # Convert relevant columns to numeric if they exist
        numeric_columns = ['open', 'high', 'low', 'close', 'percent_change', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert 'f_date' to datetime if the column exists
        if 'f_date' in df.columns:
            df['f_date'] = pd.to_datetime(df['f_date'])

        # Sort the DataFrame by date if the column exists
        if 'f_date' in df.columns:
            df_sorted = df.sort_values(by='f_date', ascending=False)

            filename = f"{symbol}.csv"
            # Save DataFrame to CSV
            path = os.path.join(os.getcwd(), 'datas/' + filename)
            df_sorted.to_csv(path, index=False)
            print(f"Data saved for {symbol} in {path}")
            print(df_sorted)
        else:
            print(f"Column 'f_date' does not exist in the DataFrame for {symbol}.")
    else:
        print(f"Failed to retrieve data for {symbol}. Status code: {response.status_code}")
        print(response.text)  # Print the response text for debugging

# List of stock symbols to retrieve data for
stocks_banking = ['NABIL', 'KBL', 'MBL', 'SANIMA', 'NICA']
stocks_finance = ['CFCL', 'GFCL', 'MFIL', 'GUFL', 'NFL']
stocks_microfinance = ['CBBL', 'DDBL', 'SKBBL', 'SMFDB', 'SWBBL', 'SMB', 'FOWAD', 'KLBSL']
stocks_life = ['ALICL', 'LICN', 'NLIC', 'CLI', 'ILI', 'SJLIC']
stocks_non_life = ['NICL', 'NIL', 'NLG', 'SICL', 'PRIN', 'HEI']
stocks_others = ['NRIC', 'NTC', 'SHIVM', 'HRL']
stocks_hydro = ['NYADI', 'RADHI', 'NHPC', 'KPCL', 'HDHPC', 'DHPL', 'API', 'AKPL', 'UNHPL']

stock_symbols = stocks_banking + stocks_finance + stocks_microfinance + stocks_life + stocks_non_life + stocks_hydro + stocks_others

# Loop through each symbol to get data
for symbol in stock_symbols:
    get_stock_data(symbol)
