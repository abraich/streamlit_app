
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


# define a function to scrape all coins non listed on binance
def get_all_coins():
    url = 'https://api.binance.com/api/v1/exchangeInfo'
    response = requests.get(url)
    data = response.json()
    coins = []
    for pair in data['symbols']:
        coins.append(pair['symbol'])
    return coins
# test the function
print(get_all_coins())