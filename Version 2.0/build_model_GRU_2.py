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

import tensorflow as tf


from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



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
        
        
        
# prepare data for ML model
# the training set represents 80% of the data given to the model
# the test represents the 20% of the data to validate the model
# Assuming x_train and y_train are your data arrays


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

x_data = x_train[:,:5]
x_stockdata=x_train[:,5:]

min_max_scaler_stockdata = preprocessing.MinMaxScaler()
min_max_scaler_data = preprocessing.MinMaxScaler()


x_stockdata_scaled = min_max_scaler_stockdata.fit_transform(x_stockdata)
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



# Build the model
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



# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 0], label='Actual Bid', color='blue')
plt.plot(predictions[:, 0], label='Predicted Bid', color='red', linestyle='--')
#plt.plot(y_test[:, 1], label='Actual Ask', color='green')
#plt.plot(predictions[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Bid Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# Assuming y_test and predictions are 2D arrays with shape [samples, 2] (for bid and ask prices)
actual_bid_prices = y_test[:, 0]
predicted_bid_prices = predictions[:, 0]

# Calculate absolute percentage error for bids, avoiding divide by zero
absolute_error_bid = np.abs(actual_bid_prices - predicted_bid_prices)
absolute_error_bid_percent = np.where(actual_bid_prices != 0, (absolute_error_bid / actual_bid_prices) * 100, 0)

# Define the range for the histogram bins (0% to 500% with 100 bins)
bins = np.linspace(0, 500, 100)

# Plot a histogram of the absolute percentage errors within the specified range
plt.figure(figsize=(12, 6))
plt.hist(absolute_error_bid_percent, bins=bins, edgecolor='k', color='blue', alpha=0.7)
plt.title('Distribution of Percentage Error for Bid Prices')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Plot a bar graph of the absolute errors for each sample
plt.figure(figsize=(12, 6))
plt.plot(absolute_error_bid, color='blue', alpha=0.7)
plt.title('Absolute Error for Bid Prices (Sample-wise)')
plt.xlabel('Sample Index')
plt.ylabel('Absolute Error')
plt.grid(True)
plt.show()


# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 1], label='Actual Ask', color='green')
plt.plot(predictions[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Ask Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()


# Assuming y_test and predictions are 2D arrays with shape [samples, 2] (for bid and ask prices)
actual_ask_prices = y_test[:, 1]
predicted_ask_prices = predictions[:, 1]

# Calculate absolute percentage error for bids, avoiding divide by zero
absolute_error_ask = np.abs(actual_ask_prices - predicted_ask_prices)
absolute_error_ask_percent = np.where(actual_ask_prices != 0, (absolute_error_ask / actual_ask_prices) * 100, 0)

# Define the range for the histogram bins (0% to 500% with 100 bins)
bins = np.linspace(0, 500, 100)

# Plot a histogram of the absolute percentage errors for asks
plt.figure(figsize=(12, 6))
plt.hist(absolute_error_ask_percent, bins=bins, edgecolor='k', color='green', alpha=0.7)
plt.title('Histogram of Absolute Percentage Error for Ask Prices')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot a bar graph of the absolute errors for each sample
plt.figure(figsize=(12, 6))
plt.plot(absolute_error_ask, color='green', alpha=0.7)
plt.title('Absolute Error for Ask Prices (Sample-wise)')
plt.xlabel('Sample Index')
plt.ylabel('Absolute Error')
plt.grid(True)
plt.show()




# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
#plt.plot(y_test[-100:, 0], label='Actual Bid', color='blue')
#plt.plot(predictions[-100:, 0], label='Predicted Bid', color='red', linestyle='--')
plt.plot(y_test[-100:, 1], label='Actual Ask', color='green')
plt.plot(predictions[-100:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Ask Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()



# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
plt.plot(y_test[-100:, 0], label='Actual Bid', color='blue')
plt.plot(predictions[-100:, 0], label='Predicted Bid', color='red', linestyle='--')
#plt.plot(y_test[:, 1], label='Actual Ask', color='green')
#plt.plot(predictions[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Bid Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()



# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:10, 0], label='Actual Bid', color='blue')
plt.plot(predictions[:10, 0], label='Predicted Bid', color='red', linestyle='--')
#plt.plot(y_test[:, 1], label='Actual Ask', color='green')
#plt.plot(predictions[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Bid Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()



# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
#plt.plot(y_test[:10, 0], label='Actual Bid', color='blue')
#plt.plot(predictions[:10, 0], label='Predicted Bid', color='red', linestyle='--')
plt.plot(y_test[:10, 1], label='Actual Ask', color='green')
plt.plot(predictions[:10, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Bid Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()



# Preprocess the new ticker's data
new_ticker = 'AMZN'  # Replace with the new ticker symbol
x_test, y_test = produceXYDataSets(new_ticker, "C", 20)

# Scale the test data using the same scalers as the training data
# Assuming min_max_scaler_stockdata and min_max_scaler_data are already fitted with training data
x_test_data = x_test[:,:5]
x_test_stockdata = x_test[:,5:]
x_test_stockdata_scaled = min_max_scaler_stockdata.transform(x_test_stockdata)
x_test_data_scaled = min_max_scaler_data.transform(x_test_data)

# Prepare the data for LSTM (reshape if necessary)
x_test_lstm = np.concatenate((np.expand_dims(x_test_stockdata_scaled, axis=2), np.repeat(np.expand_dims(x_test_data_scaled, 1), x_test_stockdata_scaled.shape[1], axis=1)), axis=2)

# Make predictions
predictions_test = model.predict(x_test_lstm)

# Inverse transform the predictions
predictions_test_original = predictions_test

# Compare predictions with actual values
# ... (You can use a similar plotting function as before to visualize the results)

# Example: Plotting the results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 0], label='Real Bid', color='blue')
plt.plot(predictions_test_original[:, 0], label='Predicted Bid', color='red', linestyle='--')
plt.plot(y_test[:, 1], label='Real Ask', color='green')
plt.plot(predictions_test_original[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title(f'Real vs Predicted Bid/Ask Prices for {new_ticker}')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()


mape = mean_absolute_percentage_error(y_test, predictions_test_original)
print("\nBid error,        Ask Error")
print(mape)


# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 0], label='Actual Bid', color='blue')
plt.plot(predictions_test_original[:, 0], label='Predicted Bid', color='red', linestyle='--')
#plt.plot(y_test[:, 1], label='Actual Ask', color='green')
#plt.plot(predictions[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Bid Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# Assuming y_test and predictions are 2D arrays with shape [samples, 2] (for bid and ask prices)
actual_bid_prices = y_test[:, 0]
predicted_bid_prices = predictions_test_original[:, 0]

# Calculate absolute percentage error for bids, avoiding divide by zero
absolute_error_bid = np.abs(actual_bid_prices - predicted_bid_prices)
absolute_error_bid_percentage = np.where(actual_bid_prices != 0, (absolute_error_bid / actual_bid_prices) * 100, 0)

# Define the range for the histogram bins (0% to 500% with 100 bins)
bins = np.linspace(0, 500, 100)

# Plot a histogram of the absolute percentage errors
plt.figure(figsize=(12, 6))
plt.hist(absolute_error_bid_percentage, bins=bins, edgecolor='k', color='blue', alpha=0.7)
plt.title('Distributiuon of Percentage Error for Bid Prices')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Plot a bar graph of the absolute  errors for each sample
plt.figure(figsize=(12, 6))
plt.plot(absolute_error_bid, color='blue', alpha=0.7)
plt.title('Absolute Error for Bid Prices (Sample-wise)')
plt.xlabel('Sample Index')
plt.ylabel('Absolute Error')
plt.grid(True)
plt.show()


# Define the range for the histogram bins (0% to 500% with 100 bins)
bins = np.linspace(0, 200, 100)

# Plot a histogram of the absolute percentage errors
plt.figure(figsize=(12, 6))
plt.hist(absolute_error_bid_percentage, bins=bins, edgecolor='k', color='blue', alpha=0.7)
plt.title('Distributiuon of Percentage Error for Bid Prices on AMZN data')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Plotting the predictions against actual values
# Assuming y_test and y_pred are 2D arrays with shape [samples, 2] (for bid and ask prices)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 1], label='Actual Ask', color='green')
plt.plot(predictions_test_original[:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title('Actual vs Predicted Ask Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()


# Assuming y_test and predictions are 2D arrays with shape [samples, 2] (for bid and ask prices)
actual_ask_prices = y_test[:, 1]
predicted_ask_prices = predictions_test_original[:, 1]


# Calculate absolute percentage error for bids, avoiding divide by zero
absolute_error_ask = np.abs(actual_ask_prices - predicted_ask_prices)
absolute_error_ask_percentage = np.where(actual_ask_prices != 0, (absolute_error_ask / actual_ask_prices) * 100, 0)

# Define the range for the histogram bins (0% to 500% with 100 bins)
bins = np.linspace(0, 500, 100)

# Plot a histogram of the absolute percentage errors for asks
plt.figure(figsize=(12, 6))
plt.hist(absolute_error_ask_percentage, bins=bins, edgecolor='k', color='green', alpha=0.7)
plt.title('Histogram of Absolute Percentage Error for Ask Prices')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot a bar graph of the absolute percentage errors for each sample
plt.figure(figsize=(12, 6))
plt.plot(absolute_error_ask, color='green', alpha=0.7)
plt.title('Absolute  Error for Ask Prices (Sample-wise)')
plt.xlabel('Sample Index')
plt.ylabel('Absolute Error')
plt.grid(True)
plt.show()


# Define the range for the histogram bins (0% to 500% with 100 bins)
bins = np.linspace(0, 200, 100)

# Plot a histogram of the absolute percentage errors for asks
plt.figure(figsize=(12, 6))
plt.hist(absolute_error_ask_percentage, bins=bins, edgecolor='k', color='green', alpha=0.7)
plt.title('Histogram of Absolute Percentage Error for Ask Prices on AMZN data')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(y_test[-100:, 0], label='Real Bid', color='blue')
plt.plot(predictions_test_original[-100:, 0], label='Predicted Bid', color='red', linestyle='--')
plt.plot(y_test[-100:, 1], label='Real Ask', color='green')
plt.plot(predictions_test_original[-100:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title(f'Real vs Predicted Bid/Ask Prices for {new_ticker}')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(y_test[-10:, 0], label='Real Bid', color='blue')
plt.plot(predictions_test_original[-10:, 0], label='Predicted Bid', color='red', linestyle='--')
plt.plot(y_test[-10:, 1], label='Real Ask', color='green')
plt.plot(predictions_test_original[-10:, 1], label='Predicted Ask', color='orange', linestyle='--')
plt.title(f'Real vs Predicted Bid/Ask Prices for {new_ticker}')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()

