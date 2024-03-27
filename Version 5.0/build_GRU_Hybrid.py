import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from PrepareOptionsData import produceXYDataSets
from RNN_model_GRU_2 import build_gru_model_hybrid
from joblib import dump
import warnings

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Setting for reproducibility
np.random.seed(42)

tickers = ['AAPL', 'AMD']
option_type = 'C'
ns_back = 20

# Initialize data storage
x_train_all, y_train_all = [], []

for ticker in tickers:
    try:
        x_train_ticker, y_train_ticker = produceXYDataSets(ticker, option_type, ns_back)
        x_train_all.append(x_train_ticker)
        y_train_all.append(y_train_ticker)
    except Exception as e:
        print(f"Failed to process ticker {ticker}: {e}")

# Concatenate the data
x_train = np.concatenate(x_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0) if isinstance(y_train_all[0], np.ndarray) else np.array(y_train_all)

# Separate option and stock data
x_option_data = x_train[:, 1:6]
x_stock_data = x_train[:, 6:]

# Initialize scalers
min_max_scaler_option = MinMaxScaler()
min_max_scaler_stock = MinMaxScaler()

# Scale data
x_option_data_scaled = min_max_scaler_option.fit_transform(x_option_data)
x_stock_data_scaled = min_max_scaler_stock.fit_transform(x_stock_data)

# Assuming each row in x_stock_data_scaled is a time step and reshaping for GRU input
x_stock_data_scaled = np.expand_dims(x_stock_data_scaled, axis=2)

# Split the data into training and testing sets
test_size = int(0.2 * len(x_train))
x_option_train, x_option_test = x_option_data_scaled[:-test_size], x_option_data_scaled[-test_size:]
x_stock_train, x_stock_test = x_stock_data_scaled[:-test_size], x_stock_data_scaled[-test_size:]
y_train, y_test = y_train_all[:-test_size], y_train_all[-test_size:]

# Check if y_train needs reshaping
if len(y_train.shape) == 1:
    # If y_train is expected to have only one target value per sample
    y_train = y_train.reshape(-1, 1)


# Build the hybrid model

# x_option_train shape  (69639, 5)
# x_stock_train shape  (69639, 20, 1)
x_option_train_expanded = np.expand_dims(x_option_train, axis=-1)
print('x_option_train_expanded shape ', x_option_train_expanded.shape[:])

print('x_stock_train shape ', x_stock_train.shape[:])

# For option data, exclude the first dimension (batch size) and keep the rest
option_data_shape = x_option_train_expanded.shape[1:]

# For stock data, it's already in the correct format (samples, timesteps, features)
# So, just pass the shape excluding the first dimension
stock_data_shape = x_stock_train.shape[1:]

# Now, use these shapes to build the model
model = build_gru_model_hybrid(option_data_shape, stock_data_shape)
model.summary()

# Train the model

# Define callbacks
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
# model_checkpoint = ModelCheckpoint('best_gru_model_hybrid.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with callbacks
history = model.fit(
    [x_option_train, x_stock_train], y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    # callbacks=[model_checkpoint]
)

# Evaluate the model
test_loss = model.evaluate([x_option_test, x_stock_test], y_test)
print(f"Test Loss: {test_loss}")

predictions = model.predict([x_option_test, x_stock_test])

# Save the model and scalers
model_name = 'split_options_GRU_Hybrid'
model.save(f'{model_name}_model.keras')

dump(min_max_scaler_option, 'min_max_scaler_Hybrid_option.joblib')
dump(min_max_scaler_stock, 'min_max_scaler_Hybrid_stock.joblib')

# Save predictions and test sets
np.save('GRU_Hybrid_predictions.npy', predictions)
np.save('GRU_Hybrid_x_option_test.npy', x_option_test)
np.save('GRU_Hybrid_x_stock_test.npy', x_stock_test)
np.save('GRU_Hybrid_y_test.npy', y_test)

# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error on Test Set: {mse}")
print(f"Mean Absolute Error on Test Set: {mae}")

def mean_absolute_percentage_error(y_true, y_pred): 
    epsilon = 1e-8  # Small number to avoid division by zero
    bid_true, ask_true = y_true[:, 0], y_true[:, 1]
    bid_pred, ask_pred = y_pred[:, 0], y_pred[:, 1]
    
    bid_error = np.mean(np.abs((bid_true - bid_pred) / (bid_true + epsilon))) * 100
    ask_error = np.mean(np.abs((ask_true - ask_pred) / (ask_true + epsilon))) * 100
    
    return [bid_error, ask_error]

mape = mean_absolute_percentage_error(y_test, predictions)
print("\nBid error, Ask Error:", mape)

