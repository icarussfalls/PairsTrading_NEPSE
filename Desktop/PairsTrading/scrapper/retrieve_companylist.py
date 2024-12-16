import requests
import json
import time
import pandas as pd
from datetime import datetime
import os

# Read the company list from a file
with open("company_list.json", "r") as file:
    companies = json.load(file)

# Iterate over all companies in the list
for company in companies:
    symbol = company["d"]  # Company symbol (e.g., ADBL, CBL)
    print(symbol)