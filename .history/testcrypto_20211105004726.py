
import requests
import json
import datetime
import time
import sys
import configparser
import argparse
import logging
import logging.handlers
import os
import traceback
import pprint
import threading
import queue
import signal
import random
import re
import math
import decimal
import csv
import copy
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceWithdrawException
#from binance.websockets import BinanceSocketManager
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceWithdrawException
# build a class for crypto trading

def transform_coin(coin):
    if '_' in coin:
        coin = coin.replace('_', '')
        coin = coin.upper()
    else:
        coin = coin.upper()
        coin = coin + 'USDT'
    return coin

# define a function to scrape  coins which will  listed on gate.io and not in binance
def scrape_coins():
    print("Scraping coins")

    # define the gate.io API URL
    gate_url = "https://data.gateio.io/api2/1/tickers"

    # define the binance API URL
    binance_url = "https://api.binance.com/api/v1/ticker/24hr"

    # get the data from gate.io
    gate_response = requests.get(gate_url, headers={'Accept': 'application/json', 'Accept-Charset': 'utf-8'})
    gate_data = json.loads(gate_response.text)

    # get the data from binance
    binance_response = requests.get(binance_url, headers={'Accept': 'application/json', 'Accept-Charset': 'utf-8'})
    binance_data = json.loads(binance_response.text)

    # define the gate.io coins
    gate_coins = []
    for coin in gate_data:
        if 'usdt' in coin:
            gate_coins.append(transform_coin(coin))

    # define the binance coins
    binance_coins = []
    for coin in binance_data:
        if 'USDT' in coin['symbol']:
            binance_coins.append(coin['symbol'])
        
    print("Gate.io coins: ", gate_coins,len(gate_coins))
    print("Binance coins: ", binance_coins,len(binance_coins))
    diff = list(set(gate_coins) - set(binance_coins))
    print("Difference: ", diff,len(diff))
    print("Scraping coins complete")
# test the scrape_coins function
scrape_coins()


# defin a function to trandform like bnb_usdt to BNBUSDT

#print("Transform coin: ", transform_coin('shib_usdt'))