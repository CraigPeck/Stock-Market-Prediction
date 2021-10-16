#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:48:57 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
#import tensorflow as tf
from AI_Train_Test_Data import *
from TechnicalIndicators import *
from createFeaturesNew import *
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
from sklearn.preprocessing import RobustScaler 
import csv  
from datetime import date, timedelta, datetime
import pandas_datareader as webreader


today = date.today()
date_today = today.strftime("%Y-%m-%d")
date_start = '2016-01-01'

# Getting NASDAQ quotes
stockname = 'GOOGLE'
symbol = 'GOOGL'
df = webreader.DataReader(
    symbol, start=date_start, end=date_today, data_source="yahoo"
)


file = pd.DataFrame({'Adj Close': df['Adj Close'], 'Adj Low' : df['Low'], 'Adj High' : df['High']})




########Convert DataFrame into an array
file = np.array(file)
#Call function createFeatures to create technical indicators from file 
features = createFeatures(file)

#Combine any chosen features such as closing price and a technical indicator into a single DataFrame 
#file_feature = pd.DataFrame({'Adj Close': features[0], 'Bollinger Band Upper': features['Bollinger_Upper'], 'Bollinger Band Lower' : features['Bollinger_Lower'], 'MACD' : features['MACD'] })
file_feature = pd.DataFrame({'Adj Close': features[0], 'Adj Low': features[1], 'Adj High' : features[2], 'MACD' : features['MACD'], 'EMA20': features['EMA20'] })



#Create the scaler to normalize the data between 0 and 1 to make model training easier
scaler = MinMaxScaler(feature_range=(0,1))
#Create a second scaler for future inverse transform on correct data set size
scaler_pred = MinMaxScaler(feature_range = (0,1))
#Scale selected data to fit model 
file_scaled = scaler.fit_transform(np.array(file_feature).reshape(-5,5))
df_adj_close = file[:,0]
np_close_scaled = scaler_pred.fit(np.array(df_adj_close).reshape(-1,1))

#Create new DataFrame from scaled data
file = pd.DataFrame(file_scaled)

#Create training and test sets using DataProcessing class 
process = DataProcessing(0.7, file)
process.gen_test(10)
process.gen_train(10)


X_train = process.X_train
Y_train = process.Y_train 
Y_train = Y_train[:,:, 0]


X_test = process.X_test
Y_test = process.Y_test 
Y_test = Y_test[:,:, 0]

#Set aside actual stock prices for later comparrison to the predicted prices
real_stock_price = scaler.inverse_transform(X_test[:, 0])
real_stock_price = real_stock_price[:,0]



#Generate the LSTM model we will use
model=Sequential()

#Define the number of neurons in the model based on the size of the input training set
n_nuerons = X_train.shape[1] * X_train.shape[2]

#Create first layer of the LSTM model with the specified input shape and number of neurons
model.add(LSTM(n_nuerons,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
#Add drop out of 20% to help avoid overfitting the model to the data 
model.add(Dropout(0.1))
model.add(LSTM(n_nuerons,return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(n_nuerons, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(64))
#Define desired output and sigmoid activation function relu 
model.add(Dense(1, activation = 'relu'))
#Define loss metric and optimizer for reweighting in the neural network 
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

#Fit the created model to the training sets and define the epochs: number of times the model iterates and reweights 
model.fit(X_train,Y_train,epochs= 50,batch_size=64,verbose=1)


train_predict=model.predict(X_train)
#Predict next day's closing price based on the generated model and test set
test_predict=model.predict(X_test)

# Mean Absolute Percentage Error (MAPE) as a means to show model performance
MAPE = np.mean((np.abs(np.subtract(Y_test, test_predict)/ Y_test))) * 100
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

#Convert the scaled values back into real prices 
train_predict=scaler_pred.inverse_transform(train_predict)
test_predict=scaler_pred.inverse_transform(test_predict)

#Plot results of the predictions against the true prices 
plt.plot(real_stock_price, color = 'black', label = 'Google Stock Price')
plt.plot(test_predict, color = 'green', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


####################### Predicting Tomorrow's Stock Price

# Get fresh data until today and create a new dataframe with only the price data
date_start2 = date.today() - timedelta(days=200)
new_df = webreader.DataReader(symbol, data_source='yahoo', start=date_start2, end=date_today)
d = pd.to_datetime(new_df.index)
new_df['Month'] = d.strftime("%m") 
new_df['Year'] = d.strftime("%Y") 
features_new = createFeaturesNew(new_df)
#new_df = pd.DataFrame({'Close': new_df['Close'], 'Bollinger Band Upper' : features_new['Bollinger_Upper'], 'Bollinger Band Lower' : features_new['Bollinger_Lower'], 'MACD': features_new['MACD']})
new_df = pd.DataFrame({'Close': new_df['Close'], 'Low' : features_new['Low'], 'High' : features_new['High'], 'MACD' : features_new['MACD'], 'EMA20': features_new['EMA20'] })



# Get the last 100 day closing price values and scale the data to be values between 0 and 1
last_100_days = new_df[-100:].values
scaler_new = MinMaxScaler(feature_range = (0,1))
last_100_days_scaled = scaler_new.fit_transform(last_100_days)

# Create an empty list and Append past 100 days
X_test_new = []
X_test_new.append(last_100_days_scaled)



# Convert the X_test data set to a numpy array and reshape the data
pred_price_scaled = model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled)

# Print last price and predicted price for the next day
price_today = round(new_df['Close'][-1], 2)
predicted_price = round(pred_price_unscaled.ravel()[0], 2)
percent = round(100 - (predicted_price * 100)/price_today, 2)

a = '+'
if percent > 0:
    a = '-'

print(f'The close price for {stockname} at {today} was {price_today}')
print(f'The predicted close price is {round(predicted_price, 1)} ({a}{percent}%)')