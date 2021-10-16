#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 19:34:49 2021

@author: craigpeck
"""

#### Binance API Call 

import numpy as np
import pandas as pd
from pandas import DataFrame
 
#import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
import quandl
from sklearn import linear_model

from binance.client import Client
client = Client(api_key = 'En3RDjfsytbaaOSZ71CssUrTcXa1SoAyQ6uMs582720YHQiprMnfjvaCQSbHlFuh' , api_secret = 'mux3dJQUljxsEmiBug9XMTnmSom9KKo8VJ3pVgNagzhLukXWoClsy0gqZgYBukni' )


# get market depth
#depth = client.get_order_book(symbol='BNBBTC')

# get historical kline data from any date range

# fetch 1 Day  klines for the last month of 2017
klines1 = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jun, 2018", "30 Dec, 2020")
klines2 = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jun, 2018", "30 Dec, 2020")
klines3 = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jun, 2018", "30 Dec, 2020")
klines4 = client.get_historical_klines("EOSUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jun, 2018", "30 Dec, 2020")

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
    
x_norm = normalize_data(x)
y = normalize_data(y)
z = normalize_data(z)
w = normalize_data(w)

x = np.array(x_norm).reshape(-1,1)


df.plot()


    
ind_train = pd.DataFrame({'y': y,'z': z, 'w': w})
yzw_train = ind_train[['y', 'z', 'w']]
    
regr = linear_model.LinearRegression()
results_reg = regr.fit(x, yzw_train)
results_reg = regr.coef_
print(results_reg)
hedgeRatio = results_reg
hedgeRatio_columns = np.array(hedgeRatio).reshape(1,-1)
hedgeRatio_ind = pd.DataFrame(hedgeRatio_columns, columns = {'yCoef', 'zCoef', 'wCoef'})


pltVal = pd.DataFrame(x_norm - hedgeRatio_ind['yCoef']*y - hedgeRatio_ind['zCoef']*z - hedgeRatio_ind['wCoef']*w)
pltVal.plot()

# Johansen test
result=vm.coint_johansen(df[['x', 'y', 'z', 'w']].values, det_order=0, k_ar_diff=1)
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

# Add IGE for Johansen test
result=vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

print(result.eig) # eigenvalues
print("Eig Vec =" ,result.evec) # eigenvectors

yport=pd.DataFrame(np.dot(df.values, result.evec[:, 0])) #  (net) market value of portfolio

ylag=yport.shift()
deltaY=yport-ylag
df2=pd.concat([ylag, deltaY], axis=1)
df2.columns=['ylag', 'deltaY']
regress_results=sm.ols(formula="deltaY ~ ylag", data=df2).fit() # Note this can deal with NaN in top row
print(regress_results.params)

halflife=-np.log(2)/regress_results.params['ylag']
print('halflife=%f days' % halflife)