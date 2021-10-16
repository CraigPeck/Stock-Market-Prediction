#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:46:29 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
#import tensorflow as tf
from AI_Train_Test_Data import *
#import pandas_datareader.data as pdr
import quandl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error



df1 = quandl.get("Wiki/GOOGL", start_date = "2016-01-01", end_date = "2018-01-02")

file = pd.DataFrame(df1['Adj. Close'])
#file = np.array(file)

#plt.plot(file)

scaler = MinMaxScaler(feature_range=(0,1))
file_scaled = scaler.fit_transform(np.array(file).reshape(-1,1))

file = pd.DataFrame(file_scaled)

process = DataProcessing(0.5, file)
process.gen_test(10)
process.gen_train(10)


X_train = process.X_train
Y_train = process.Y_train 


X_test = process.X_test
Y_test = process.Y_test 

real_stock_price = scaler.inverse_transform(X_test[:,1])




model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,Y_train,epochs=100,batch_size=64,verbose=1)


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


plt.plot(real_stock_price, color = 'black', label = 'Google Stock Price')
plt.plot(test_predict, color = 'green', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



