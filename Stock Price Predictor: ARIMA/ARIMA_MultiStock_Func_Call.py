#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:42:11 2021

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


df1 = quandl.get("Wiki/GOOGL", start_date = "2016-01-01", end_date = "2017-02-02")
df2 = quandl.get("Wiki/TSLA", start_date = "2016-01-01", end_date = "2017-02-02")
df3 = quandl.get("Wiki/AAPL", start_date = "2016-01-01", end_date = "2017-02-02")
df4 = quandl.get("Wiki/FB", start_date = "2016-01-01", end_date = "2017-02-02")
df5 = quandl.get("Wiki/AMZN", start_date = "2016-01-01", end_date = "2017-02-02")
df6 = quandl.get("Wiki/NFLX", start_date = "2016-01-01", end_date = "2017-02-02")
df7 = quandl.get("Wiki/CRM", start_date = "2016-01-01", end_date = "2017-02-02")
df8 = quandl.get("Wiki/NVDA", start_date = "2016-01-01", end_date = "2017-02-02")

d_tot = pd.DataFrame({'Asset 1': df1['Adj. Close'], 'Asset 2': df2['Adj. Close'], 'Asset 3': df3['Adj. Close'], 'Asset 4': df4['Adj. Close'], 'Asset 5': df5['Adj. Close'], 'Asset 6': df6['Adj. Close'], 'Asset 7': df7['Adj. Close'], 'Asset 8': df8['Adj. Close']})
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
Predicted_Stock_Returns = pd.DataFrame({'Stock 1': Total_Predicted_Returns[:,0], 'Stock 2': Total_Predicted_Returns[:,1], 'Stock 3': Total_Predicted_Returns[:,2], 'Stock 4': Total_Predicted_Returns[:,3], 'Stock 5': Total_Predicted_Returns[:,4], 'Stock 6': Total_Predicted_Returns[:,5], 'Stock 7': Total_Predicted_Returns[:,6], 'Stock 8': Total_Predicted_Returns[:,7]})
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
actual_ret_long = []
actual_ret_short = []           

for p in range(len(df_dly_ret_test)):
    
    
    #for r in range(d_array_lngth):
        
    Day_Predicted_Returns = Total_Predicted_Returns[p,:]
        
    t_p = np.percentile(Day_Predicted_Returns, 80, axis = 0, interpolation = 'nearest')
    tp_where = np.where(Day_Predicted_Returns == np.percentile(Day_Predicted_Returns, 80, axis = 0, interpolation = 'nearest'))
    b_p = np.percentile(Day_Predicted_Returns, 20, axis = 0, interpolation = 'nearest')
    bp_where = np.where(Day_Predicted_Returns == np.percentile(Day_Predicted_Returns, 20, axis = 0, interpolation = 'nearest'))
     
    tp_where = int(tp_where[0]) 
    bp_where = int(bp_where[0])
    
    actual_long = df_dly_ret_test[p, (tp_where - 1)]
    actual_short = df_dly_ret_test[p, (bp_where - 1)]
    actual_short_pos = (-1)*(actual_short)
    
    actual_ret_long.append(actual_long)
    actual_ret_short.append(actual_short_pos)
    

    
total_daily_returns = pd.DataFrame({'Long': actual_ret_long, 'Short': actual_ret_short})

port_ret = total_daily_returns.sum(axis = 1)

port_ret.plot()


plt.show()