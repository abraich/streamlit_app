
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
class CryptoTrader:
    def __init__(self, config_file, symbol, strategy, strategy_parameters, log_level, log_file, log_max_size, log_file_count, log_format, log_format_file, log_format_console):
        self.config_file = config_file
        self.symbol = symbol
        self.strategy = strategy
        self.strategy_parameters = strategy_parameters
        self.log_level = log_level
        self.log_file = log_file
        self.log_max_size = log_max_size
        self.log_file_count = log_file_count
        self.log_format = log_format
        self.log_format_file = log_format_file
        self.log_format_console = log_format_console
        self.logger = self.setup_logging()
        self.logger.info("Initializing Binance API")
        self.client = Client(self.api_key, self.api_secret)
        self.logger.info("Initializing Binance API - Done")
        self.logger.info("Initializing Binance WebSocket")
        #self.ws = BinanceSocketManager(self.client)
        self.logger.info("Initializing Binance WebSocket - Done")
        self.logger.info("Initializing Strategy")
        self.strategy = strategy(self.logger, self.client, self.ws, self.symbol, self.strategy_parameters)
        self.logger.info("Initializing Strategy - Done")
        self.logger.info("Initializing Order Book")
        self.order_book = self.client.get_order_book(symbol=self.symbol)
        self.logger.info("Initializing Order Book - Done")
        self.logger.info("Initializing Trade History")
                self.trade_history = self.client.get_my_trades(symbol=self.symbol)
        self.logger.info("Initializing Trade History - Done")
        self.logger.info("Initializing Account Info")
        self.account_info = self.client.get_account()
        self.logger.info("Initializing Account Info - Done")
        self.logger.info("Initializing Account Balances")
        self.account_balances = self.client.get_account_balance()
        self.logger.info("Initializing Account Balances - Done")
        self.logger.info("Initializing Account Positions")
        self.account_positions = self.client.get_position_info()
        self.logger.info("Initializing Account Positions - Done")
        self.logger.info("Initializing Open Orders")
        self.open_orders = self.client.get_open_orders(symbol=self.symbol)
        self.logger.info("Initializing Open Orders - Done")
        self.logger.info("Initializing Order History")
        self.order_history = self.client.get_all_orders(symbol=self.symbol)
        self.logger.info("Initializing Order History - Done")
        self.logger.info("Initializing Open Orders")
        self.open_orders = self.client.get_open_orders(symbol=self.symbol)
        self.logger.info("Initializing Open Orders - Done")
        self.logger.info("Initializing Order History")
        self.order_history = self.client.get_all_orders(symbol=self.symbol)
        self.logger.info("Initializing Order History - Done")
        self.logger.info("Initializing Order Book")
        self.order_book = self.client.get_order_book(symbol=self.symbol)
        self.logger.info("Initializing Order Book - Done")
                self.logger.info("Initializing Trade History")
        self.trade_history = self.client.get_my_trades(symbol=self.symbol)
        self.logger.info("Initializing Trade History - Done")
        self.logger.info("Initializing Account Info")
        self.account_info = self.client.get_account()
        self.logger.info("Initializing Account Info - Done")
        self.logger.info("Initializing Account Balances")
        self.account_balances = self.client.get_account_balance()
        self.logger.info("Initializing Account Balances - Done")
        self.logger.info("Initializing Account Positions")
        self.account_positions = self.client.get_position_info()
        self.logger.info("Initializing Account Positions - Done")
        self.logger.info("Initializing Order Book")
        self.order_book = self.client.get_order_book(symbol=self.symbol)
        self.logger.info("Initializing Order Book - Done")
        self.logger.info("Initializing Trade History")
        self.trade_history = self.client.get_my_trades(symbol=self.symbol)
        self.logger.info("Initializing Trade History - Done")
        self.logger.info("Initializing Account Info")
        self.account_info = self.client.get_account()
        self.logger.info("Initializing Account Info - Done")
        self.logger.info("Initializing Account Balances")
        self.account_balances = self.client.get_account_balance()
        self.logger.info("Initializing Account Balances - Done")
        self.logger.info("Initializing Account Positions")
        self.account_positions = self.client.get_position_info()
        
        