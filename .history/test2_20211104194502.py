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

class Binance : 
    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.client = Client(self.key, self.secret)
        self.strategy = None
        self.coins = None
    
    def get_coins(self):
        return self.coins
    def set_coins(self, coins):
        self.coins = coins
    def get_data(self):
        return self.client.get_all_tickers()
    def get_data_by_coin(self, coin):
        return self.client.get_ticker(symbol=coin)
    def buy_coin(self, coin, quantity, price):
        return self.client.order_limit_buy(symbol=coin, quantity=quantity, price=price)
    def sell_coin(self, coin, quantity, price):
        return self.client.order_limit_sell(symbol=coin, quantity=quantity, price=price)
    def get_price_coin(self, coin):
        return self.client.get_ticker(symbol=coin)

# define a strategy class that will be used to buy/sell coins
def Strategy :
    
    