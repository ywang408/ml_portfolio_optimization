import os
from pathlib import Path
import pandas as pd
import yfinance as yf
import time
from tqdm import tqdm

# vars for fetching data
api = Path('alpha_vantage_api.txt').read_text()
base_url = 'https://www.alphavantage.co/query?function='
interval = 'interval=monthly'
datatype = 'datatype=csv'
indicator = ['FEDERAL_FUNDS_RATE', 'CPI','INFLATION_EXPECTATION',\
             'CONSUMER_SENTIMENT', 'UNEMPLOYMENT']
maturity = ['3month','2year','5year','10year']

# change dir
os.chdir('../data')

# a test
# url = 'https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&datatype=csv&apikey=A2KW5JUMDQAKFTGN'
# data = pd.read_csv(url)
# data.to_csv('test.csv')
# print(data)

# Alpha Vange allows five requests per minute, so we download data seperately
print("Start downloading Treasury Yield Data...")
for i in indicator:
    url = f'{base_url}{i}&{interval}&{datatype}&apikey={api}'
    data = pd.read_csv(url)
    data.to_csv(f'{i.lower()}.csv')
    print(f'{i} downloaded')

print("Economic Indicators's Download Will Continue In 60 Seconds, Don't Stop The Program")
for i in tqdm(range(30)):
    time.sleep(2)
url = f'{base_url}TREASURY_YIELD&{interval}&{datatype}&apikey={api}'
for t in maturity:
    url = f'{url}&maturity={t}'
    data = pd.read_csv(url)
    data.to_csv(f'treasury{t}.csv')
    print(f'treasury{t} downloaded')

# get companies' tickers in spy
spy500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
stock_data_table = pd.read_html(spy500url)
tickers = stock_data_table[0].Symbol.to_list()

# download companies prices data
print("Start downloading Stocks' Data...")
stock_data = yf.download(tickers=tickers, start="1990-01-01", end="2022-09-01")
stock_data.to_pickle("./stocks.pkl")
