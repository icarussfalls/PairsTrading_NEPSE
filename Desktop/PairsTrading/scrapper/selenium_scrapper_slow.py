#!/usr/bin/env python
# coding: utf-8

from selenium import webdriver
import os
import time
from fake_useragent import UserAgent
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# Chrome options
options = webdriver.ChromeOptions()

class Scrapper():
    def __init__(self, url):
        ua = UserAgent()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'user-agent={ua.random}')
        options.add_argument('--headless')  # Run in headless mode for speed
        
        # Block only images and other non-essential resources
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values": {
                "cookies": 2, "images": 2, 
                "plugins": 2, "popups": 2, "geolocation": 2, 
                "notifications": 2, "auto_select_certificate": 2, "fullscreen": 2, 
                "mouselock": 2, "mixed_script": 2, "media_stream": 2, 
                "media_stream_mic": 2, "media_stream_camera": 2, 
                "protocol_handlers": 2, "ppapi_broker": 2, "automatic_downloads": 2, 
                "midi_sysex": 2, "push_messaging": 2, "ssl_cert_decisions": 2, 
                "metro_switch_to_desktop": 2, "protected_media_identifier": 2, 
                "app_banner": 2, "site_engagement": 2, "durable_storage": 2
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        service = Service("/Users/icarus/Downloads/chromedriver-mac-arm64 2/chromedriver")  # Your specific path
        self.driver = webdriver.Chrome(service=service, options=options)
        self.url = url
        self.driver.get(url)
        # Navigate to the history tab
        WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab"]')))
        self.driver.find_element("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab"]').click()
        self.driver.implicitly_wait(2)

    def df(self):
        max_attempts = 3  # Set a maximum retry count

        for attempt in range(max_attempts):
            try:
                # Wait for the table to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr'))
                )

                # Extract all rows from the table
                rows = self.driver.find_elements(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr')

                # Initialize list to collect data
                data = []

                # Iterate over rows and extract the required data
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) >= 9:  # Ensure there are enough columns
                        data.append({
                            'Date': cols[1].text,
                            'LTP': cols[2].text,
                            'Change': cols[3].text,
                            'High': cols[4].text,
                            'Low': cols[5].text,
                            'Open': cols[6].text,
                            'Quantity': cols[7].text,
                            'Turnover': cols[8].text
                        })

                # Create DataFrame from the collected data
                df = pd.DataFrame(data)

                # Break out of the retry loop if no exception occurred
                return df
            except StaleElementReferenceException as e:
                print(f"Stale element encountered, retrying... (attempt {attempt})")
                time.sleep(2)  # Wait before retrying
            except Exception as e:  # Catch other potential exceptions
                print(f"Failed to scrape table: {e}")
                break  # Exit loop if maximum retries are reached

        # Return None if all retries fail
        return None  # Indicate failure




    def datas(self):
        print("Started scraping for:", self.url)
        data = []
        data.append(self.df())

        while True:
            try:
                # Wait until the "Next Page" button is clickable
                k = WebDriverWait(self.driver, 20).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@title = "Next Page"]'))
                )
                
                # Perform the click action once the button is visible
                actions = ActionChains(self.driver)
                actions.move_to_element(k).click().perform()

                time.sleep(2)

                # Scrape the new data after moving to the next page
                scrap = self.df()
                data.append(scrap)

            except Exception as e:
                print("No more pages. Finished. Error:", e)
                break

        # Concatenate all the scraped data into a single DataFrame
        data = pd.concat(data, axis=0)
        
        # Remove duplicate dates again after concatenating all pages
        final_data = data.drop_duplicates(subset='Date', keep='first')
        
        return final_data
    
    def close(self):
        self.driver.quit()

def save_datas(symbol):
    url = f'https://merolagani.com/CompanyDetail.aspx?symbol={symbol}'
    scrapper = Scrapper(url)
    try:
        data = scrapper.datas()
        filename = f"{symbol}.csv"
        path = os.path.join(os.getcwd(), filename)
        data.to_csv(path, index=False)
        print(f"Data saved for {symbol} in {path}")
    finally:
        scrapper.close()

stocks_banking = ['NABIL', 'KBL', 'MBL', 'SANIMA', 'NICA']
stocks_finance = ['CFCL', 'GFCL', 'MFIL', 'GUFL', 'NFL']
stocks_microfinance = ['CBBL', 'DDBL', 'SKBBL', 'SMFDB', 'SWBBL', 'SMB', 'FOWAD', 'KLBSL' ]
stocks_life = ['ALICL', 'LICN', 'NLIC', 'CLI', 'ILI', 'SJLIC']
stocks_non_life = ['NICL', 'NIL', 'NLG', 'SICL', 'PRIN', 'HEI']
stocks_others = ['NRIC', 'NTC']

pending = ['SANIMA', 'CFCL', 'NFL', 'DDBL', 'FOWAD', 'ILI', 'NIL', 'NTC']

if __name__ == '__main__':

    stock_symbols = stocks_banking + stocks_finance + stocks_microfinance + stocks_life + stocks_non_life + stocks_others
    stock_symbols = ['NTC']

    max_workers = 3  # Adjust the number of workers based on your system

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(save_datas, stock_symbols)