import math

import numpy as np
import quandl
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pandas as pd

def main():
    df = quandl.get('WIKI/GOOGL')
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
    df.dropna(inplace=True)

    print(df.head())
    print(df.tail())

    # Splitting and training inputs (everything except the label) to predict output (label or future close price)
    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])
    X = preprocessing.scale(X)

    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # classifier = svm.SVR(kernal='poly')
    classifier = LinearRegression()

    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)

    print(f"With {accuracy:.3f} accuracy, I can predict stock price {forecast_out_days} days into the future")

if __name__ == '__main__':
    main()

