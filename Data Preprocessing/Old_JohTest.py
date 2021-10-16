#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:33:58 2021

@author: craigpeck
"""

import quandl
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros, ones, flipud, log
from numpy.linalg import inv, eig, cholesky as chol
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen

#### Get Apple and Microsoft Data 

df_x = quandl.get("Wiki/AAL", start_date = "2017-01-01", end_date = "2019-01-02")
df_y = quandl.get("Wiki/LUV", start_date = "2017-01-01", end_date = "2019-01-02")

#### APPLE 

df_x.head()
df_x.tail()
df_x.describe()
df_x.index
df_x.columns
ts = df_x['Close'][-10:]


#df_x['Close'].plot(grid = True)
#plt.show()



#### Microsoft

df_y.head()
df_y.tail()
df_y.describe()
df_y.index
df_y.columns
ts = df_y['Close'][-10:]


#df_y['Close'].plot(grid = True)
#plt.show()




#### Cointegration Test: Johansen Test

df = pd.DataFrame({'x': df_x['Close'], 'y': df_y['Close']})
df.plot(grid = True)
plt.show()



ctest = coint_johansen(df, 0, 1)

print("\nnormalized eigenvector 0\n", ctest.evec[:,0] / ctest.evec[:,0][0])
print("\nnormalized eigenvector 1\n", ctest.evec[:,1] / ctest.evec[:,1][0])
print("\ntest statistics\n", ctest.lr1[0], ctest.lr1[1])
print("\ncritical values\n", ctest.cvt[0], ctest.cvt[1])
print("\neig\n", ctest.eig)
print("\nevec\n", ctest.evec)
print("\nlr1\n", ctest.lr1)
print("\nlr2\n", ctest.lr2)
print("\ncvt\n", ctest.cvt)
print("\ncvm\n", ctest.cvm)
print("\nind\n", ctest.ind)
