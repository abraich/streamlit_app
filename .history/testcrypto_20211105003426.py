
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


# define a function to scrape  coins which will  listed on gate.io and not in binance
def scrape_coins():
    print("Scraping coins")
    # read the gate.io API key
    config = configparser.ConfigParser()
    config.read('config.ini')
    gate_key = config.get('gateio', 'key')
    gate_secret = config.get('gateio', 'secret')

    # read the binance API key
    config = configparser.ConfigParser()
    config.read('config.ini')
    binance_key = config.get('binance', 'key')
    binance_secret = config.get('binance', 'secret')

    # define the gate.io API URL
    gate_url = "https://data.gateio.io/api2/1/tickers"

    # define the binance API URL
    binance_url = "https://api.binance.com/api/v1/ticker/24hr"

    # define the binance ws API URL
    binance_ws_url = "wss://stream.binance.com:9443/stream?streams=!miniTicker@arr"

    # define the gate.io API URL
    gate_url = "https://data.gateio.io/api2/1/tickers"

    # define the binance API URL
    binance_url = "https://api.binance.com/api/v1/ticker/24hr"

    # define the binance ws API URL
    binance_ws_url = "wss://stream.binance.com:9443/stream?streams=!miniTicker@arr"

    # define the gate.io API URL
    gate_url = "https://data.gateio.io/api2/1/tickers"

    # define the binance API URL