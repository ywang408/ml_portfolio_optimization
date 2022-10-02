from glob import glob
import pandas as pd
import yfinance as yf

def indicator_data_cleaning():
    os.chdir('../data')

    # init a dataframe only contains data of cpi
    indicator_df = pd.read_csv('cpi.csv')
    indicator_df.timestamp = pd.to_datetime(indicator_df.timestamp)
    indicator_df = indicator_df.set_index('timestamp')
    indicator_df = indicator_df.drop(indicator_df.columns[0],axis=1)
    indicator_df.index.name = 'date'
    indicator_df.columns = ['cpi']
    indicator_df = indicator_df.sort_index(ascending=True)
    indicator_df = indicator_df[indicator_df.index>'1990-01-01']

    # iteratively add indicators' data to the dataframe
    indicator_files = glob('*.csv')
    indicator_files.remove('cpi.csv')
    for i in indicator_files:
        tmp = pd.read_csv(i)
        tmp.timestamp = pd.to_datetime(tmp.timestamp)
        tmp = tmp.set_index('timestamp')
        tmp = tmp.drop(tmp.columns[0],axis=1)
        tmp.index.name = 'date'
        tmp.columns = [i.replace('.csv','')]
        tmp = tmp.sort_index(ascending=True)
        tmp = tmp[tmp.index>'1990-01-01']
        indicator_df = indicator_df.join(tmp)

    indicator_df.to_pickle('indicators.pkl')