
# Pairs Trading Strategy for NEPSE

This repository implements a pairs trading strategy tailored for the Nepal Stock Exchange (NEPSE). Below is a step-by-step guide to using the scripts and understanding their functionality.

## Steps to Run the Strategy

1. **Retrieve Company List**  
   Run the `retrieve_company_results.py` script located in the `scrapper` directory. This retrieves the list of companies.

2. **Download Company Data**  
   Execute the `merolagani.py` script to fetch stock data for the companies. The downloaded data will be stored in the `data` folder.

3. **Preprocess Data**  
   Use the `preprocess_data.py` script to filter stocks and check for stationarity. The results will be saved in the `cointegration_results.csv` file.

4. **Backtest the Strategy**  
   Run the `pairstradingwithOU.py` script to perform backtesting on the selected pairs. The backtest results are automatically saved in `results/cointegrated_pairs_results.csv`.

5. **Filter and Save Best Signals**  
   The `save_best_signals.py` script can be used to sort signals by Sharpe ratio, win rate, or half-life and save the best-performing ones.

6. **Generate Live Signals**  
   Run the `signals.py` script located in the `buy_and_exit` directory to generate live trading signals. This will create a `pairs_trading_signals.csv` file with **buy**, **hold**, and **exit** signals.
