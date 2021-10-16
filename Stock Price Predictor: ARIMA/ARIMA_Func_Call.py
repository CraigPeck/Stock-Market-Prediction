#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:55:15 2021

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
from ARIMA_OPTIMIZED_Def import *


df1 = quandl.get("Wiki/LUV", start_date = "2016-01-01", end_date = "2016-04-02")
df2 = quandl.get("Wiki/TSLA", start_date = "2016-01-01", end_date = "2016-04-02")
df3 = quandl.get("Wiki/MSFT", start_date = "2016-01-01", end_date = "2016-04-02")
df4 = quandl.get("Wiki/GOOGL", start_date = "2016-01-01", end_date = "2016-04-02")

d_tot = pd.DataFrame({'Asset 1': df1['Adj. Close'], 'Asset 2': df2['Adj. Close'], 'Asset 3': df3['Adj. Close'], 'Asset 4': df4['Adj. Close']})
d_array = np.array(d_tot)
d_array_transpose = d_array.T
d_array_lngth = d_array_transpose
d_array_lngth = len(d_array_lngth)
df_dly_ret = pd.DataFrame(d_tot.pct_change()[1:]) 
df_AdjClose_test = d_tot[0:int(len(d_tot)*0.3)]
df_dly_ret_test = df_dly_ret[0:int(len(df_dly_ret)*0.3)]
df_dly_ret_test = np.array(df_dly_ret_test)

d_ind = len(d_array)
d_range = range(d_ind)

Pred_Ret = []
Predicted_Returns = []

for i in range(d_array_lngth):
   
    df_col = d_array[:,i]

    for t in range(len(d_array)):
    
        
        df = pd.DataFrame(df_col)
        model = ARIMA_Optimized(df)
        model = np.array(model)
        model = np.reshape(model, (1, len(model)))
    
    
        Pred_Ret.append(model)
        
        
        break
    
Pred_Ret_Array = np.vstack(Pred_Ret)
Pred_Ret_Array = Pred_Ret_Array.transpose()
Predicted_Returns.append(Pred_Ret_Array.transpose())
   
    
        
 
Total_Predicted_Returns = np.vstack(Predicted_Returns)  
Total_Predicted_Returns = Total_Predicted_Returns.transpose() 
#Total_Predicted_Returns = np.delete(Total_Predicted_Returns, 0, 1)
Predicted_Stock_Returns = pd.DataFrame({'Stock 1': Total_Predicted_Returns[:,0], 'Stock 2': Total_Predicted_Returns[:,1], 'Stock 3': Total_Predicted_Returns[:,2], 'Stock 4': Total_Predicted_Returns[:,3]})
total_cum_long_ret = []
total_cum_short_ret = []   

dr_array_lngth = Total_Predicted_Returns.shape
dr_array_lngth = len(dr_array_lngth) 
lngth_index = len(Total_Predicted_Returns)      
 

total_day_ret = []
ret_long_placeholder = np.zeros(df_dly_ret_test.shape) 
ret_short_placeholder = np.zeros(df_dly_ret_test.shape)
ret_placeholder = np.zeros(df_dly_ret_test.shape)
ret_dly_placeholder = []             

for p in range(len(df_dly_ret_test)):
    
    
    for r in range(d_array_lngth):
        
        Day_Predicted_Returns = Total_Predicted_Returns[:,r]
        
        if Day_Predicted_Returns[r]>0:
            ret_long = df_dly_ret_test[p,r]
            ret_placeholder[p,r] = ret_long
            
        elif Day_Predicted_Returns[r]<0:
            ret_short = (df_dly_ret_test[p,r])
            ret_placeholder[p,r] = ret_short
      
ret_dly_placeholder = pd.DataFrame({'Stock 1 Ret': ret_placeholder[:,0], 'Stock 2 Ret': ret_placeholder[:,1], 'Stock 3 Ret': ret_placeholder[:,2], 'Stock 4 Ret': ret_placeholder[:,3]})
        

ret_dly_placeholder = np.array(ret_dly_placeholder)
top_percentile = []
bottom_percentile = []  


for g in range(len(Total_Predicted_Returns)):
    
     t_p = np.percentile(ret_dly_placeholder[g,:], 75, axis = 0)
     tp_where = np.where(np.percentile(ret_dly_placeholder[g,:], 75, axis = 0))
     b_p = np.percentile(ret_dly_placeholder[g,:], 25, axis = 0)
     bp_where = np.where(np.percentile(ret_dly_placeholder[g,:], 25, axis = 0))
     b_p = (-1)*(b_p)
     
     top_percentile.append(t_p)
     bottom_percentile.append(b_p)
    
total_daily_returns = pd.DataFrame({'Long': top_percentile, 'Short':bottom_percentile})

port_ret = total_daily_returns.sum(axis = 1)


cum_daily_returns = (1+port_ret).cumprod()

cum_daily_returns.plot()

plt.show()
    

    
      
        
    
