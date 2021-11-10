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
from matplotlib.finance import candlestick_ohlc

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
    