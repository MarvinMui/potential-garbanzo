import math

import numpy as np
import quandl
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

def main():
    style.use('ggplot')

    df = quandl.get('WIKI/GOOGL')
    # df.to_csv('googl.csv')
    # df = pd.read_csv('googl.csv', index_col='Date', parse_dates=True)

    # some different ways of exploring or dissecting a dataframe
    # print(df.head())
    # print(df.iloc[:5])
    # print(df.columns)
    # print(df.dtypes)
    # print(df.index)
    # print(df.loc[df.index <= '2004-08-25'])

    # For linear regression, not all features are distinct or useful
    useful_col = ['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']
    df = df[useful_col]

    # more useful than an arbitrary stock price is the volatility or change in a day
    # the high low percent change
    df['hl pct'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
    # daily final movement
    df['pct change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

    useful_col = ['Adj. Close', 'hl pct', 'pct change', 'Adj. Volume']
    df = df[useful_col]\


    # Best choice to use future close price as label (predicted output) because current close price is inherently
    # a feature, or closely linked to other features.
    forecast_col = "Adj. Close"
    df.fillna(-99999, inplace=True)

    forecast_out_days = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out_days)

    print(df.head())
    print(df.tail())

    # Splitting and training inputs (everything except the label) to predict output (label or future close price)
    # Also saving the unpredicted set of values for attempting to predict the future unknown close prices
    X = np.array(df.drop(['label'], axis=1))
    X = preprocessing.scale(X)
    X_unpredected = X[-forecast_out_days:]
    X = X[:-forecast_out_days]

    df.dropna(inplace=True)
    y = np.array(df['label'])

    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # classifier = svm.SVR(kernal='poly')
    classifier = LinearRegression()

    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)

    print(f"With {accuracy:.3f} accuracy, I can predict stock price {forecast_out_days} days into the future")

    forcast = classifier.predict(X_unpredected)

    print(f"\nNext {forecast_out_days} days of google prices: \n", forcast)

    df["Forecast"] = np.NAN
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    day_in_sec = 24 * 60 * 60
    next_unix = last_unix + day_in_sec

    for i in forcast:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += day_in_sec
        df.loc[next_date] = [np.NAN for _ in range(len(df.columns) - 1)] + [i]

    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

if __name__ == '__main__':
    main()

