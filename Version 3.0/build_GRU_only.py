from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# gather datasets for each ticker defined
from PrepareOptionsData import produceXYDataSets
import warnings


tickers = ['AAPL', 'AMD']
option_type = 'C'
#  represents the number of previous data points (or "steps back") to consider when training
ns_back = 20

# declare a dict to contain the x, y data for each ticker
stock_data_dict = {}

x_train_all = []
y_train_all = []

for ticker in tickers:
    try:
       
        x_train_ticker, y_train_ticker = produceXYDataSets(ticker, option_type, ns_back)
        x_train_all.append(x_train_ticker)
        y_train_all.append(y_train_ticker)
    
        # fill the dict with the ticker name as the key and then the associated x and y
        stock_data_dict[ticker] = (x_train_ticker, y_train_ticker)

    except Exception as e:
        print(f"Failed to process ticker {ticker}: {e}")
        
print(stock_data_dict)


# prepare data for ML model
# the training set represents 80% of the data given to the model
# the test represents the 20% of the data to validate the model
# x_train and y_train are your data arrays


# Concatenate the lists into numpy arrays
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# concat the data from each ticker into a continous stream array
x_train = np.concatenate(x_train_all, axis=0)

# Check if y_train_all is a list of arrays
if isinstance(y_train_all, list) and all(isinstance(elem, np.ndarray) for elem in y_train_all):
    # Concatenate all arrays in the list
    y_train_all = np.concatenate(y_train_all, axis=0)
else:
    # If y_train_all is not a list of arrays, handle it accordingly
    raise ValueError("y_train_all is not a list of NumPy arrays")

# Now y_train_all is a single NumPy array
print("Shape of y_train_all:", y_train_all.shape)

# Continue with the rest of your code

#y_train = np.concatenate(y_train_all, axis=0)

# Extract contract names
x_contract_names = x_train[:, 0]

# Adjust the slicing indices for x_data and x_stock_data
# Assuming the next 5 elements after the contract name are still your features
x_data = x_train[:, 1:6]  # Adjust indices as needed
x_stock_data = x_train[:, 6:]  # Assuming stock prices start from the 7th element


min_max_scaler_stockdata = preprocessing.MinMaxScaler()
min_max_scaler_data = preprocessing.MinMaxScaler()


x_stockdata_scaled = min_max_scaler_stockdata.fit_transform(x_stock_data)
#print(x_stockdata_scaled)

x_data_scaled = min_max_scaler_data.fit_transform(x_data)
#print(x_data_scaled)

# x_stockdata_scaled is temporal data with shape [samples, time steps]
# Reshape x_stockdata_scaled to [samples, time steps, features] if it's not already in 3D
if len(x_stockdata_scaled.shape) == 2:
    # Assuming each row is a time step and you have only one feature per time step
    x_stockdata_scaled = np.expand_dims(x_stockdata_scaled, axis=2)

# Now x_stockdata_scaled should be 3D
# Repeat x_data_scaled to match the time steps in x_stockdata_scaled
# and x_data_scaled are additional features that we will append to each time step
x_data_scaled_repeated = np.repeat(np.expand_dims(x_data_scaled, 1), x_stockdata_scaled.shape[1], axis=1)

# Concatenate along the last dimension
x_lstm = np.concatenate((x_stockdata_scaled, x_data_scaled_repeated), axis=2)

# Convert lists to NumPy arrays
x_lstm = np.array(x_lstm)

# Check the shapes
print("Shape of x_lstm:", x_lstm.shape)

# Split the data into training and testing sets
# Calculate the number of samples for the test set (20% of total samples)
test_size = int(0.2 * len(x_lstm))

# Split the data into training and testing sets
x_train = x_lstm[:-test_size]
x_test = x_lstm[-test_size:]
y_train = y_train_all[:-test_size]
y_test = y_train_all[-test_size:]

'''print('x_train')
print(x_train)

print('x_test')
print(x_test)

print('y_train')
print(y_train)

print('y_test')
print(y_test)'''



# Build the old model
from RNN_model_GRU_2 import build_gru_model

from sklearn.impute import SimpleImputer

# Check if y_train needs reshaping
if len(y_train.shape) == 1:
    # If y_train is expected to have only one target value per sample
    y_train = y_train.reshape(-1, 1)


# Check again for NaN values
if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
    print("NaN values are still present")
else:
    print("NaN values have been removed")

model_name = 'split_options_GRU'
model = build_gru_model(input_shape=(x_train.shape[1], x_train.shape[2]))

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

print("\nEvaluate on the testing data:")

# Evaluate the model on the test set
test_loss = model.evaluate(x_test, y_test)

# Predictions on the test set
predictions = model.predict(x_test)

# Save the model
model.save(model_name+'_model.h5')

print(f"Test Loss: {test_loss}")


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error on Test Set: {mse}")
print(f"Mean Absolute Error on Test Set: {mae}")


# Function to calculate Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    bid_true = y_true[:][0]
    ask_true = y_true[:][1]
    
    bid_pred = y_pred[:][0]
    ask_pred = y_pred[:][1]
    
    bid_error = np.mean(np.abs((bid_true-bid_pred)/bid_true))*100
    ask_error = np.mean(np.abs((ask_true-ask_pred)/ask_true))*100
    
    
    return [bid_error, ask_error]


mape = mean_absolute_percentage_error(y_test, predictions)
print("\nBid error,        Ask Error")
print(mape)

import numpy as np

# After obtaining predictions from your model
np.save('GRU_predictions.npy', predictions)
np.save('GRU_x_test', x_test)
np.save('GRU_y_test.npy', y_test)
