#! /bin/python3

import datetime

import pandas.io.data as web
from pandas import HDFStore
from ystockquote import get_historical_prices
import urllib2
from odo import odo
import pandas as pd
from pandas import DataFrame
from numpy import array

from utils import get_sp500


def get_historical(symbols, start, end):
    quotes = dict()
    for symbol in symbols:
        print(symbol)
        try:
            # quotes[symbol] = get_historical_prices(symbol, start, end)
            quotes[symbol] = DataFrame(
                get_historical_prices(symbol, start, end)).swapaxes(0, 1)
        except urllib2.HTTPError:
            print("{} Failed".format(symbol))
    return quotes


def get_stock_data(start='2010-01-01',
                   end='2015-01-01',

                   save=False):

    print('Fetching s&p500 names')
    symbols, names, industry, _ = get_sp500(start[:4])
    print('...Done\n')
    # symbols = symbols[:10]
    print('Fetching data from yahoo finance')
    quotes = get_historical(symbols, start, end)
    quotes = pd.Panel(quotes).swapaxes('items', 'minor')
    quotes = quotes.dropna(axis=2)
    print('...Done\n')

    print('Done getting stocks')

    if save:
        with HDFStore('test.hdf5') as store:
            for key in quotes:
                quote = quotes[key]
                # print([key for key, test in quote.isnull().sum().astype(bool).iteritems() if test])
                # quote.drop([key for key, test in quote.isnull().sum().astype(bool).iteritems() if test], axis=1, inplace=True)

                if key == 'Volume':
                    quote = quote.astype(int)
                    # store[key] = quote
                else:
                    quote = quote.astype(float)
                store[key.replace(' ', '_')] = quote

if __name__ == '__main__':
    get_stock_data(save=True)
