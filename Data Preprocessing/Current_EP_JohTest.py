#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:35:19 2021

@author: craigpeck
"""

# Using the CADF test for cointegration

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
import quandl


df_x = quandl.get("Wiki/AAL", start_date = "2017-01-01", end_date = "2020-01-02")
df_y = quandl.get("Wiki/UAL", start_date = "2017-01-01", end_date = "2020-01-02")
df = pd.DataFrame({'x': df_x['Close'], 'y': df_y['Close']})

df.plot()
df.plot.scatter(x='x', y='y')

results=sm.ols(formula="x ~ y", data=df[['x', 'y']]).fit()
print(results.params)
hedgeRatio=results.params[1]
print('hedgeRatio=%f' % hedgeRatio)

pltVal = pd.DataFrame((df['x']-hedgeRatio*df['y']))
pltVal.plot()


# Johansen test
result=vm.coint_johansen(df[['x', 'y']].values, det_order=0, k_ar_diff=1)
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
print(result.evec) # eigenvectors

yport=pd.DataFrame(np.dot(df.values, result.evec[:, 0])) #  (net) market value of portfolio

ylag=yport.shift()
deltaY=yport-ylag
df2=pd.concat([ylag, deltaY], axis=1)
df2.columns=['ylag', 'deltaY']
regress_results=sm.ols(formula="deltaY ~ ylag", data=df2).fit() # Note this can deal with NaN in top row
print(regress_results.params)

halflife=-np.log(2)/regress_results.params['ylag']
print('halflife=%f days' % halflife)