import os
from glob import glob
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import yfinance as yf
import talib


def download_eco_indic():
    if not os.path.exists('../data/eco_indicators'):
        os.makedirs('../data/eco_indicators')
        print("economic indicators' directory made")
        print("-------------------------------------")
    api = Path('alpha_vantage_api.txt').read_text()
    base_url = 'https://www.alphavantage.co/query?function='
    interval = 'interval=monthly'
    datatype = 'datatype=csv'
    indicator = ['FEDERAL_FUNDS_RATE', 'CPI', 'INFLATION_EXPECTATION', \
                 'CONSUMER_SENTIMENT', 'UNEMPLOYMENT']
    maturity = ['3month', '2year', '5year', '10year']

    # Alpha Vange allows five requests per minute, so we download data separately
    print("Start downloading Treasury Yield Data...")
    for i in indicator:
        url = f'{base_url}{i}&{interval}&{datatype}&apikey={api}'
        data = pd.read_csv(url)
        data.to_csv(f'../data/eco_indicators/{i.lower()}.csv')
        print(f'{i} downloaded')

    print("Economic Indicator's Download Will Continue In 60 Seconds, Don't Stop The Program")
    for i in tqdm(range(30)):
        time.sleep(2)
    # continue download
    url = f'{base_url}TREASURY_YIELD&{interval}&{datatype}&apikey={api}'
    for t in maturity:
        url = f'{url}&maturity={t}'
        data = pd.read_csv(url)
        data.to_csv(f'../data/eco_indicators/treasury{t}.csv')
        print(f'treasury{t} downloaded')


def eco_indic_processing():
    eco_indic_files = glob('../data/eco_indicators/*.csv')
    print(eco_indic_files)
    if len(eco_indic_files) < 7:
        download_eco_indic()
    eco_indic_files = glob('../data/eco_indicators/*.csv')

    # init a dataframe only contains data of cpi
    eco_indic_df = pd.read_csv('../data/eco_indicators/cpi.csv')
    eco_indic_df.timestamp = pd.to_datetime(eco_indic_df.timestamp)
    eco_indic_df = eco_indic_df.set_index('timestamp')
    eco_indic_df = eco_indic_df.drop(eco_indic_df.columns[0], axis=1)
    eco_indic_df.index.name = 'date'
    eco_indic_df.columns = ['cpi']
    eco_indic_df = eco_indic_df.sort_index(ascending=True)
    eco_indic_df = eco_indic_df[eco_indic_df.index > '1990-01-01']

    # iteratively add indicators' data to the dataframe
    eco_indic_files.remove('../data/eco_indicators/cpi.csv')
    for i in eco_indic_files:
        tmp = pd.read_csv(i)
        tmp.timestamp = pd.to_datetime(tmp.timestamp)
        tmp = tmp.set_index('timestamp')
        tmp = tmp.drop(tmp.columns[0], axis=1)
        tmp.index.name = 'date'
        tmp.columns = [i.replace('.csv', '')]
        tmp = tmp.sort_index(ascending=True)
        tmp = tmp[tmp.index > '1990-01-01']
        eco_indic_df = eco_indic_df.join(tmp)
    eco_indic_df.to_pickle('../data/eco_indicators/eco_indicator.pkl')


def stock_indic_processing(ticker):
    if not os.path.exists('../data/stocks/tech_indic/'):
        os.makedirs('../data/stocks/tech_indic/')
        print("stocks' directory made")

    stock_df = yf.download(ticker, start="1993-01-29", end='2022-08-30', progress=False)
    stock_df = stock_df.dropna()
    stock_df['return'] = np.log(stock_df['Adj Close'] / stock_df['Adj Close'].shift(1))
    stock_df = stock_df.dropna()

    # technical indicators dataframe calculated in pandas
    stock_indic_df = pd.DataFrame([], index=stock_df.index)
    stock_indic_df['sma5'] = stock_df['Close'].rolling(5).mean()
    stock_indic_df['sma30'] = stock_df['Close'].rolling(30).mean()
    stock_indic_df['std15'] = stock_df['Close'].rolling(15).std()
    stock_indic_df['std30'] = stock_df['Close'].rolling(30).std()
    stock_indic_df['H-L'] = stock_df['High'] - stock_df['Low']
    stock_indic_df['C-O'] = stock_df['Close'] - stock_df['Open']
    stock_indic_df['ewm15'] = stock_df['Close'].ewm(15).mean()

    # technical indicators calculated by ti_lab
    stock_indic_df['rsi'] = talib.RSI(stock_df['Close'].values)
    stock_indic_df['will_r'] = talib.WILLR(stock_df['High'].values, stock_df['Low'].values, stock_df['Close'].values)
    stock_indic_df['sar'] = talib.SAR(stock_df['High'].to_numpy(), np.array(stock_df['Low']))
    stock_indic_df['adx'] = talib.ADX(stock_df['High'].to_numpy(), stock_df['Low'].to_numpy(),
                                      stock_df['Close'].to_numpy())

    stock_indic_df = stock_indic_df.dropna()
    stock_df = stock_df[stock_df.index >= stock_indic_df.index[0]]
    stock_df.to_pickle(f'../data/stocks/{ticker}.pkl')
    stock_indic_df.to_pickle(f'../data/stocks/tech_indic/{ticker}_indic.pkl')


if __name__ == '__main__':
    # eco_indic_processing()
    spy500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    stock_data_table = pd.read_html(spy500url)
    tickers = stock_data_table[0].Symbol.to_list()
    # download companies prices data
    print("Start downloading Stocks' Data...")
    for t in tqdm(tickers):
        try:
            stock_indic_processing(t)
        except Exception as e:
            print("There is an issue with ticker: {} and we are passing it".format(t))
            pass
