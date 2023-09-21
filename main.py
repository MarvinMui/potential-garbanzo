import quandl
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
    df['hl pct'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
    # daily final movement
    df['pct change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

    useful_col = ['Adj. Close', 'hl pct', 'pct change', 'Adj. Volume']
    df = df[useful_col]
    print(df.head())


if __name__ == '__main__':
    main()

