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


key = 'LbiDrvyqPMg5N75IfFxlNvWD3Nuar0W4KugQQCJjcgmSESdRsITIbJ6nRNT64VOj'
secret = 'ebu79QafzJoFM3qLdmriVN0KTJ2iiyVrCnAqfJhIz4jGT1ZAqOZ7C665FKZ5w8Js'

class Binance : 
    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.client = Client(self.key, self.secret)
        self.strategy = None
        self.coins = None
    # get account info for non-zero balance in spot account
    def get_account_info(self):
        balances =  self.client.get_account()['balances']
        balances_nonzero = [balance for balance in balances if balance['free'] != '0.00000000']
        df = pd.DataFrame(balances_nonzero)
        return df        
    def get_coins(self):
        return self.coins
    def set_coins(self, coins):
        self.coins = coins
    def get_data(self):
        return self.client.get_all_tickers()
    def get_data_by_coin_df(self, coin, interval,date1,date2):
        colum_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(self.client.get_historical_klines(symbol=coin, interval=interval, start_str=date1, end_str=date2), columns=colum_names)
        # convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    def get_data_by_coin_df_all(self, coin, interval):
        colum_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(self.client.get_historical_klines(symbol=coin, interval=interval, start_str='1 day ago UTC', end_str='now UTC',columns=colum_names))
        # convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
        

    def buy_coin(self, coin, quantity, price):
        return self.client.order_limit_buy(symbol=coin, quantity=quantity, price=price)
    def sell_coin(self, coin, quantity, price):
        return self.client.order_limit_sell(symbol=coin, quantity=quantity, price=price)
    def get_price_coin(self, coin):
        return self.client.get_symbol_ticker(symbol=coin)['price']



# define a class for using bianance api to sell/buy  of a coin with a grid strategy
#Grid trading is when orders are placed above and below a set price, creating a grid of orders at incrementally increasing and decreasing prices. Grid trading is most commonly associated with the foreign exchange market. Overall the technique seeks to capitalize on normal price volatility in an asset by placing buy and sell orders at certain regular intervals above and below a predefined base price. 

class Grid_Trading :
    
    
        

B = Binance(key, secret) 
print(B.get_account_info())
print(B.get_data_by_coin_df('SHIBUSDT', Client.KLINE_INTERVAL_1MINUTE, '1 day ago UTC', 'now UTC'))
print(B.get_price_coin('BTCUSDT'))