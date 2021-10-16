#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:34:22 2021

@author: craigpeck
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import quandl
from itertools import chain
import pmdarima as pm
from ARIMA_FuncDef_LongOnly import ARIMA_Optimized_Long

from binance.client import Client
client = Client(api_key = 'En3RDjfsytbaaOSZ71CssUrTcXa1SoAyQ6uMs582720YHQiprMnfjvaCQSbHlFuh' , api_secret = 'mux3dJQUljxsEmiBug9XMTnmSom9KKo8VJ3pVgNagzhLukXWoClsy0gqZgYBukni' )


klines1 = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jun, 2018", "30 Dec, 2019")

df = pd.DataFrame(klines1, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])

df = df['Close'].astype(float)






ARIMA_Optimized_Long(df)