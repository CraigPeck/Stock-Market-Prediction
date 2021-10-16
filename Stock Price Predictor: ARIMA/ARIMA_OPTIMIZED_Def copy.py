#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:39:21 2021

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



def ARIMA_Optimized(df):

    df_logRet = np.log(df/df.shift(1))
    df_AdjClose = np.array(df)
    df_dly_ret = pd.DataFrame(df.pct_change()[1:]) 
    df_AdjClose_test = df_AdjClose[0:int(len(df_AdjClose)*0.3)]
    df_dly_ret_test = df_dly_ret[0:int(len(df_dly_ret)*0.3)]
    df_dly_ret_test = np.array(df_dly_ret_test)


    # plt.figure()
    # lag_plot(df['Open'], lag=3)
    # plt.title('TESLA Stock - Autocorrelation plot with lag = 3')
    # plt.show()

    # plt.plot(df['log return'])
    # #plt.xticks(np.arange(0,1259, 200))
    # plt.title("TESLA stock price over time")
    # plt.xlabel("time")
    # plt.ylabel("price")
    # plt.show()

    train_data, test_data = df_logRet[0:int(len(df)*0.7)], df_logRet[int(len(df)*0.7):]

    training_data = train_data.values
    training_data = np.delete(training_data, 0)

    test_data = test_data.values
    test_data = np.delete(test_data, 0)

    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)

    param_pred = pm.auto_arima(history, start_p=1, start_q=1, 
                           test = 'adf',
                           max_p=6, max_q=6, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)


    param_output = param_pred.order

    for time_point in range(N_test_observations):
        model = ARIMA(history, order= (param_output) )
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    
    model_val = np.array(model_predictions)
    N_test_observations = (len(test_data) - 1)

    cumret = []


    for t in range(N_test_observations):
        if model_val[t] > 0:
            ret1 = df_dly_ret_test[t]
            cumret.append(ret1)
        elif model_val[t] <0:
            ret2 = (-1)*(df_dly_ret_test[t])
            cumret.append(ret2)
        
        

    cumret = pd.DataFrame(cumret) 
    port_ret =   cumret.sum(axis = 1)
        
    cum_dly_return = (1 + port_ret).cumprod()

    cum_dly_return.plot()

    plt.show()

    #MSE_error = mean_squared_error(test_data, model_predictions)
    #print('Testing Mean Squared Error is {}'.format(MSE_error))