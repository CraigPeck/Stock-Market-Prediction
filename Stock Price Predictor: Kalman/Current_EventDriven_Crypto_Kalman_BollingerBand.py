#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:27:14 2021

@author: craigpeck
"""

# Kalman Filter Mean Reversion Strategy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm
import quandl


from binance.client import Client
client = Client(api_key = 'En3RDjfsytbaaOSZ71CssUrTcXa1SoAyQ6uMs582720YHQiprMnfjvaCQSbHlFuh' , api_secret = 'mux3dJQUljxsEmiBug9XMTnmSom9KKo8VJ3pVgNagzhLukXWoClsy0gqZgYBukni' )


# get market depth
#depth = client.get_order_book(symbol='BNBBTC')

# get historical kline data from any date range

# fetch 1 Day  klines for the last month of 2017
klines1 = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "30 Dec, 2020")
klines2 = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "30 Dec, 2020")


df_x = pd.DataFrame(klines1, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])
df_y = pd.DataFrame(klines2, columns = ['Open time','Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume', '1', '2','3','4'])

df_x['Close'] = df_x['Close'].astype(float)
df_y['Close'] = df_y['Close'].astype(float)

df = pd.DataFrame({'x': df_x['Close'], 'y': df_y['Close']})

x=df['x']
y=df['y']


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

x = normalize_data(x)
y = normalize_data(y)

x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.

delta=0.0001 # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).

yhat=np.full(y.shape[0], np.nan) # measurement prediction
e=yhat.copy()
Q=yhat.copy()

# For clarity, we denote R(t|t) by P(t). Initialize R, P and beta.
R=np.zeros((2,2))
P=R.copy()
beta=np.full((2, x.shape[0]), np.nan)
Vw=delta/(1-delta)*np.eye(2)
Ve=0.001

# Initialize beta(:, 1) to zero
beta[:, 0]=0

# Given initial beta and R (and P)
for t in range(len(y)):
    if t > 0:
        beta[:, t]=beta[:, t-1]
        R=P+Vw
            
    yhat[t]=np.dot(x[t, :], beta[:, t])
#    print('FIRST: yhat[t]=', yhat[t])
    
    Q[t]=np.dot(np.dot(x[t, :], R), x[t, :].T)+Ve
#    print('Q[t]=', Q[t])

    # Observe y(t)
    e[t]=y[t]-yhat[t] # measurement prediction error
#    print('e[t]=', e[t])
#    print('SECOND: yhat[t]=', yhat[t])

    
    K=np.dot(R, x[t, :].T)/Q[t] #  Kalman gain
#    print(K)
    
    beta[:, t]=beta[:, t]+np.dot(K, e[t]) #  State update. Equation 3.11
#    print(beta[:, t])
    
 
    P=R-np.dot(np.outer(K, x[t, :]), R) # Thanks to Matthias for chaning np.dot -> np.outer!

#    print(R)

# plt.plot(beta[0, :])
# plt.plot(beta[1, :])
# plt.plot(e[2:])
# plt.plot(np.sqrt(Q[2:]))

longsEntry=e < (-np.sqrt(Q)*0.1)
longsExit =e > 0

shortsEntry=e >(np.sqrt(Q)*0.1)
shortsExit =e < 0

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
positions=pd.DataFrame(np.tile(numUnits.values, [1, 2]) * ts.add_constant(-beta[0,:].T)[:, [1,0]] *df.values) #  [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
(np.cumprod(1+ret)-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))