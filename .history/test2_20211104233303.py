# define a class for using bianance api to get data and sell/buy list of coins with a defined strategy
# import binance api client
from binance.client import Client
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
# plotly credentials
username = 'abraich.admission'
api_key = 'tWorpay6eL6dWxU6vcIO'
py.sign_in(username, api_key)

key = 'LbiDrvyqPMg5N75IfFxlNvWD3Nuar0W4KugQQCJjcgmSESdRsITIbJ6nRNT64VOj'
secret = 'ebu79QafzJoFM3qLdmriVN0KTJ2iiyVrCnAqfJhIz4jGT1ZAqOZ7C665FKZ5w8Js'


class Binance:
    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.client = Client(self.key, self.secret)
        self.strategy = None
        self.coins = None
    # get account info for non-zero balance in spot account

    def get_account_info(self):
        balances = self.client.get_account()['balances']
        balances_nonzero = [
            balance for balance in balances if balance['free'] != '0.00000000']
        df = pd.DataFrame(balances_nonzero)
        return df

    def get_coins(self):
        return self.coins

    def set_coins(self, coins):
        self.coins = coins

    def get_data(self):
        return self.client.get_all_tickers()

    def get_data_by_coin_df(self, coin, interval, date1, date2):
        colum_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                       'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(self.client.get_historical_klines(
            symbol=coin, interval=interval, start_str=date1, end_str=date2), columns=colum_names)
        # convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df

    def get_data_by_coin_df_all(self, coin, interval):
        colum_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                       'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(self.client.get_historical_klines(
            symbol=coin, interval=interval, start_str='1 day ago UTC', end_str='now UTC'), columns=colum_names)
        # convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df

    def buy_coin(self, coin, quantity, price):
        return self.client.order_limit_buy(symbol=coin, quantity=quantity, price=price)

    def sell_coin(self, coin, quantity, price):
        return self.client.order_limit_sell(symbol=coin, quantity=quantity, price=price)

    def get_price_coin(self, coin):
        return self.client.get_symbol_ticker(symbol=coin)['price']


# define a class for using bianance api to sell/buy  of a coin with a Exponential Moving Average  indicator
# input : coin, interval, date1, date2
# todo : use Exponential Moving Average   strategy to buy/sell coin
# output : buy/sell list
class EMA_strategy:
    def __init__(self, coin, interval, date1, date2, binance):
        self.binance = binance
        self.coin = coin
        self.interval = interval
        self.date1 = date1
        self.date2 = date2
        self.df = None
        self.ema_list = []
        self.buy_list = []
        self.sell_list = []
        self.strategy = 'EMA'

    def get_data(self):
        self.df = self.binance.get_data_by_coin_df(
            self.coin, self.interval, self.date1, self.date2)
        # convert close price to float
        self.df['close'] = self.df['close'].astype(float)
        return self.df

    def get_ema_list(self, n):
        df = self.df
        df['close'] = self.df['close'].astype(float)

        df['ema'] = df['close'].ewm(span=n, min_periods=n, adjust=False).mean()
        self.ema_list = df['ema'].tolist()
        return self.ema_list

    def get_buy_list(self):
        ema_list = self.get_ema_list(5)
        df = self.df
        df['close'] = self.df['close'].astype(float)
        df['buy_list'] = (df['close'] > ema_list)
        self.buy_list = df['buy_list'].tolist()
        return self.buy_list

    def get_sell_list(self):
        ema_list = self.get_ema_list(5)
        df = self.df
        df['sell_list'] = (df['close'] < ema_list)
        self.sell_list = df['sell_list'].tolist()
        return self.sell_list
    
    # plot ema prediction and close price with plotly without using matplotlib
    # plot also buy/sell moment
    def plot_ema_prediction(self):
        df = self.df
        df['close'] = self.df['close'].astype(float)
        df['ema'] = self.df['ema'].astype(float)
        trace1 = go.Scatter(x=df['open_time'], y=df['ema'],
                            mode='lines', name='EMA')
        trace2 = go.Scatter(x=df['open_time'], y=df['close'],
                            mode='lines', name='Close')
        trace3 = go.Scatter(x=df['open_time'], y=self.get_buy_list(),
                            mode='markers', name='Buy')
        trace4 = go.Scatter(x=df['open_time'], y=self.get_sell_list(),
                            mode='markers', name='Sell')
        data = [trace1, trace2, trace3, trace4]
        layout = go.Layout(title='EMA prediction',
                           xaxis=dict(title='Date'),
                           yaxis=dict(title='Price'))
        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig, filename='EMA prediction')
        
    # input: coin , futur interval
    # output : ema predection 
    def get_ema_prediction(self, coin, interval):
        df = self.binance.get_data_by_coin_df_all(coin, interval)
        df['close'] = df['close'].astype(float)
        df['ema'] = df['close'].ewm(span=5, min_periods=5, adjust=False).mean()
        return df['ema'].tolist()
     








B = Binance(key, secret)
"""
print(B.get_account_info())
print(B.get_data_by_coin_df('SHIBUSDT', Client.KLINE_INTERVAL_1MINUTE, '1 day ago UTC', 'now UTC'))
print(B.get_price_coin('SHIBUSDT'))
"""
# test EMA strategy
E = EMA_strategy('SHIBUSDT', Client.KLINE_INTERVAL_1MINUTE,
                 '1 day ago UTC', 'now UTC', B)
print(E.get_data())
E.get_ema_list(5)
print(E.get_buy_list())
print(E.get_sell_list())


E.plot_ema_prediction()

#print(E.get_ema_prediction('SHIBUSDT', Client.KLINE_INTERVAL_5MINUTE))