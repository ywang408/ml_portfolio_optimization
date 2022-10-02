import os
from pathlib import Path
import pandas as pd
import time

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
for i in indicator:
    url = f'{base_url}{i}&{interval}&{datatype}&apikey={api}'
    data = pd.read_csv(url)
    data.to_csv(f'{i.lower()}.csv')
    print(f'{i} downloaded')

print("Download will continue in 60 seconds, don't stop the program")
time.sleep(60)
url = f'{base_url}TREASURY_YIELD&{interval}&{datatype}&apikey={api}'
for t in maturity:
    url = f'{url}&maturity={t}'
    data = pd.read_csv(url)
    data.to_csv(f'treasury{t}.csv')
    print(f'treasury{t} downloaded')