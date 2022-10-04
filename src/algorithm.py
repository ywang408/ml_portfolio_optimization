import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import preprocessing, svm
from pypfopt import risk_models
from pypfopt import expected_returns
import os


def determine_regime(df):
    features = {}
    sum_std = 0
    for i in range(4):
        tmp_state = df[df["state"] == i].mean()
        tmp_return = tmp_state['return']
        tmp_std = tmp_state['std15']
        sum_std += tmp_std
        features[i] = [tmp_return, tmp_std]
    for i in range(4):
        if features[i][0] > 0:
            if features[i][1] > sum_std / 4:
                features[i].append('bull2')
            else:
                features[i].append('bull1')
        else:
            if features[i][1] > sum_std / 4:
                features[i].append('cris')
            else:
                features[i].append('bear')
    curr_state = df['state'][-1]
    return features[curr_state][-1]


def find_curr_regime(date):
    # create path
    if not os.path.exists('../data/regime'):
        os.makedirs('../data/regime')
        print("economic regime directory made")
    # read data
    spy = pd.read_pickle("../data/stocks/spy.pkl")
    spy_indic_df = pd.read_pickle("../data/stocks/tech_indic/spy_indic.pkl")
    date = pd.to_datetime(date)

    # clean moving average column and select
    spy_indic_df['sma5'] = (spy_indic_df['sma5'] - spy_indic_df['sma5'].shift(1)) / spy_indic_df['sma5']
    sel_cols = ['sma5', 'std15', 'H-L', 'C-O', 'rsi']
    sel_df = spy_indic_df[sel_cols].dropna()
    spy = spy[spy.index >= sel_df.index[0]]

    # init regime dataframe
    regime_df = pd.DataFrame([], index=spy.index)
    regime_df = regime_df[regime_df.index >= date]

    states = []
    # update regime dataframe
    for i in regime_df.index:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(sel_df[sel_df.index <= i])
        # init kmeans model
        n_clusters = 4
        km = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    n_init=n_clusters,
                    max_iter=300,
                    random_state=0)
        y = km.fit_predict(X)
        tmp_df = sel_df[sel_df.index <= i].copy(deep=True)
        tmp_df['Adj Close'] = spy['Adj Close']
        tmp_df['return'] = spy['return']
        tmp_df['state'] = y
        states.append(determine_regime(tmp_df))
    regime_df['state'] = states
    regime_df.to_pickle("../data/regime/regime.pkl")
    return regime_df


def select_stock(ticker, date):
    indic_df_path = f'../data/stocks/tech_indic/{ticker}_indic.pkl'
    indic_df = pd.read_pickle(indic_df_path)
    y = np.where(indic_df['sma30'].shift(-1) > indic_df['sma30'], 1, 0)
    split = indic_df[indic_df.index <= date].shape[0]
    X_train, X_test = indic_df.iloc[:split, :], indic_df.iloc[split:, :]
    y_train, y_test = y[:split], y[split:]

    try:
        # standardize
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        return predictions
    except:
        print(f'{ticker} error!')


def create_stock_pool(date):
    date = pd.to_datetime(date)
    indic_path = glob('../data/stocks/tech_indic/*.pkl')
    indic_path.remove('../data/stocks/tech_indic/spy_indic.pkl')

    # create list of tickers
    stocks = pd.read_pickle("../data/regime/all_stocks.pkl")
    stocks = stocks.columns.to_list()

    # create dataframe of stock pool
    spy_indic_df = pd.read_pickle("../data/stocks/tech_indic/spy_indic.pkl")
    pool_df = pd.DataFrame([], index=spy_indic_df.index)
    pool_df = pool_df[pool_df.index >= date]

    # iteratively determine buy or not buy one stock
    for s in tqdm(stocks):
        tmp_pred = select_stock(s, date)
        pool_df[s] = tmp_pred

    pool_df.to_pickle('../data/regime/pool_df.pkl')
    return pool_df


def estimate(stocks, pool, date, window):
    # find stock pool
    tmp = pool[pool.index == date]
    sel_tickers = tmp[tmp == 1].dropna(axis=1).columns.to_list()

    # process stock data
    sel_df = stocks.loc[:, sel_tickers]
    sel_df = sel_df[sel_df.index <= date]
    sel_df = sel_df.tail(window)

    # estimate
    mu = expected_returns.mean_historical_return(sel_df)
    S = risk_models.risk_matrix(sel_df, method='ledoit_wolf_constant_correlation')

    return mu, S


if __name__ == '__main__':
    # find_curr_regime('2018')
    # test = create_stock_pool('2018')
    # print(test.head())
    pass
