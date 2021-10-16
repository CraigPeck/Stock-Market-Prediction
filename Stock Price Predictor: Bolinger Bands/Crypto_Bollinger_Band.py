#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:37:00 2021

@author: craigpeck
"""


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm
import quandl
from sklearn import linear_model

from binance.client import Client
client = Client(api_key = 'En3RDjfsytbaaOSZ71CssUrTcXa1SoAyQ6uMs582720YHQiprMnfjvaCQSbHlFuh' , api_secret = 'mux3dJQUljxsEmiBug9XMTnmSom9KKo8VJ3pVgNagzhLukXWoClsy0gqZgYBukni' )


# get market depth
#depth = client.get_order_book(symbol='BNBBTC')

# get historical kline data from any date range
# fetch 1 Day  klines for the last month of 2017
klines1 = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "30 Dec, 2020")
klines2 = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "30 Dec, 2020")
klines3 = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "30 Dec, 2020")
klines4 = client.get_historical_klines("BNBUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "30 Dec, 2020")

df_x = pd.DataFrame(klines1, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])
df_y = pd.DataFrame(klines2, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])
df_z = pd.DataFrame(klines3, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])
df_w = pd.DataFrame(klines4, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])

df_x['Close'] = df_x['Close'].astype(float)
df_y['Close'] = df_y['Close'].astype(float)
df_z['Close'] = df_z['Close'].astype(float)
df_w['Close'] = df_w['Close'].astype(float)

df = pd.DataFrame({'x': df_x['Close'], 'y': df_y['Close'], 'z': df_z['Close'], 'w': df_w['Close']})

print(df)
    

x=df['x']
y=df['y']
z=df['z']
w=df['w']       

def normalize_data(x):
    min_x = x.min()
    max_x = x.max()
    
    norm_x = (x - min_x)/(max_x - min_x)
    
    return norm_x
    
def normalize_data(y):
    min_y = y.min()
    max_y = y.max()
    
    norm_y = (y - min_y)/(max_y - min_y)
    
    return norm_y

def normalize_data(z):
    min_z = z.min()
    max_z = z.max()
    
    norm_z = (z - min_z)/(max_z - min_z)
    
    return norm_z

def normalize_data(w):
    min_w = w.min()
    max_w = w.max()
    
    norm_w = (w - min_w)/(max_w - min_w)
    
    return norm_w
    
x = normalize_data(x)
y = normalize_data(y)
z = normalize_data(z)
w = normalize_data(w)

x = np.array(x).reshape(-1,1)




lookback=20
hedgeRatio=np.full(df.shape[0], np.nan)
for t in np.arange(lookback, len(hedgeRatio)):
    # regress_results=sm.ols(formula="x ~ y", data=df[(t-lookback):t]).fit() # Note this can deal with NaN in top row
    # hedgeRatio[t-1]=regress_results.params[1]
    # print(regress_results.params)
    x_train = x[(t-lookback):t]
    y_train = y[(t-lookback):t]
    z_train = z[(t-lookback):t]
    w_train = w[(t-lookback):t]
    
    ind_train = pd.DataFrame({'y_train': y_train,'z_train': z_train, 'w_train': w_train})
    yzw_train = ind_train[['y_train', 'z_train', 'w_train']]
    
    regr = linear_model.LinearRegression()
    results = regr.fit(x_train, yzw_train)
    results = regr.coef_
    #print('Coefficients: \n', regr.coef_)
  
    hedgeRatio[t-1]=results[1]
    #print(hedgeRatio)
   

yport=np.sum(ts.add_constant(-hedgeRatio)[:, [1,0,0,0]]*df, axis=1)
yport.plot()

# Bollinger band strategy
entryZscore=1
exitZscore=0

MA=yport.rolling(lookback).mean()
MSTD=yport.rolling(lookback).std()
zScore=(yport-MA)/MSTD

longsEntry=zScore < -entryZscore
longsExit =zScore > -entryZscore

shortsEntry=zScore > entryZscore
shortsExit =zScore < exitZscore

numUnitsLong=np.zeros(longsEntry.shape)
numUnitsLong[:]=np.nan

numUnitsShort=np.zeros(shortsEntry.shape)
numUnitsShort[:]=np.nan

numUnitsLong[0]=0
numUnitsLong[longsEntry]=1
numUnitsLong[longsExit]=0
numUnitsLong=pd.DataFrame(numUnitsLong)
numUnitsLong.fillna(method='ffill', inplace=True)

numUnitsShort[0]=0
numUnitsShort[shortsEntry]=-1
numUnitsShort[shortsExit]=0
numUnitsShort=pd.DataFrame(numUnitsShort)
numUnitsShort.fillna(method='ffill', inplace=True)

numUnits=numUnitsLong+numUnitsShort
positions=pd.DataFrame(np.tile(numUnits.values, [1, 4]) * ts.add_constant(-hedgeRatio)[:, [1,0,0,0]] *df.values) #  [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
pd.DataFrame((np.cumprod(1+ret)-1)).plot()

print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))