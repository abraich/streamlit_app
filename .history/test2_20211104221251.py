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
    def get_account(self):
        return self.client.get_account()
    def get_coins(self):
        return self.coins
    def set_coins(self, coins):
        self.coins = coins
    def get_data(self):
        return self.client.get_all_tickers()
    def get_data_by_coin(self, coin, interval):
        d = self.client.get_historical_klines(symbol=coin, interval=interval, start_str='1 day ago UTC')
        columns =  d
        df = pd.DataFrame(d, columns=columns)
        return df 
    def buy_coin(self, coin, quantity, price):
        return self.client.order_limit_buy(symbol=coin, quantity=quantity, price=price)
    def sell_coin(self, coin, quantity, price):
        return self.client.order_limit_sell(symbol=coin, quantity=quantity, price=price)
    def get_price_coin(self, coin):
        return self.client.get_ticker(symbol=coin)




# define a strategy buy and hold class that will be used to buy/sell coins using Bianance class
class Strategy_Buy_and_Hold :
    def __init__(self, binance, coin, quantity, start_date, end_date):
        self.binance = binance
        self.coin = coin
        self.quantity = quantity
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.profits = None
        self.start_coin_price = None
        self.end_coin_price = None
    
    def get_coin(self):
        return self.coin
    def get_quantity(self):
        return self.quantity
    def get_start_date(self):
        return self.start_date
    def get_end_date(self):
        return self.end_date
    def get_prices(self):
        return self.prices
    def get_profits(self):
        return self.profits
    def get_start_coin_price(self):
        return self.start_coin_price
    def get_end_coin_price(self):
        return self.end_coin_price
    def set_start_coin_price(self, start_coin_price):
        self.start_coin_price = start_coin_price
    def set_end_coin_price(self, end_coin_price):
        self.end_coin_price = end_coin_price
    def get_data(self):
        self.binance.set_coins([self.coin])
        self.prices = self.binance.get_data()
        self.prices = pd.DataFrame(self.prices)
        self.prices = self.prices.set_index('symbol')
        self.prices = self.prices.loc[self.coin]
        self.prices = self.prices.to_frame().transpose()
        self.prices = self.prices.reset_index()
        self.prices = self.prices.rename(columns={'index':'date'})
        self.prices = self.prices.set_index('date')
    


B = Binance(key, secret) 
B.set_coins(['BTCUSDT', 'SHIBUSDT'])
B.get_data()
print(B.get_data_by_coin('SHIBUSDT',interval='1m'))
