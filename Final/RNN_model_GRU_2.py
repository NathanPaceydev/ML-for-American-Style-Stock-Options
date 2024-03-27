import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, LayerNormalization, Concatenate, StringLookup
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the RNN model with GRU using the functional API
def build_gru_model(input_shape):
    # could split inputs into two time series and non-time series 
    # two seperate inputs that I could feed
    inputs = Input(shape=input_shape)
    
    #play around with dropout
    # units (50)
    # GRU layers with return_sequences=True to pass sequences to the next layer
    gru1 = GRU(50, return_sequences=True, recurrent_dropout=0.1)(inputs) # also check tanh activation could help normalization
    gru2 = GRU(50, return_sequences=False, recurrent_dropout=0.1)(gru1)
    
    # Dropout layer for regularization
    dropout = Dropout(0.2)(gru2) #might not need
    
    # Dense layers for predictions
    
    # might also want a flatten layer 
    dense1 = Dense(25, activation='relu')(dropout) # could duplicate this line (plY with layer)
    outputs = Dense(2, activation='linear')(dense1)  # Predicting bid and ask prices
    
    model = Model(inputs=inputs, outputs=outputs) # can make inputs an array of different data 
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, LeakyReLU
from tensorflow.keras.regularizers import l2

def build_gru_model_hybrid(option_data_shape, stock_data_shape):
    # Define inputs for each type of data
    inputOptionData = Input(shape=option_data_shape, name='x_option_train_expanded')
    inputStockData = Input(shape=stock_data_shape, name='x_stock_train')

    # Option data processing with normalization
    #x0_norm = LayerNormalization()(inputOptionData)
    #x0 = layers.Flatten()(x0_norm)
    x0 = layers.Flatten()(inputOptionData)
    
    x0 = Dense(32, activation="relu")(x0)
    x0_option_dense_output = Dense(8, activation="relu", name='option_dense_output')(x0)

    # Stock data processing with GRU and normalization
    #x1_norm = LayerNormalization()(inputStockData)
    #x1 = Bidirectional(GRU(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)))(x1_norm)
    x1 = Bidirectional(GRU(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)))(inputStockData)
    
    x1 = Dropout(0.3)(x1)  # Increased dropout
    x1 = layers.Flatten()(x1)
    x1_stock_gru_output = Dense(8, activation=LeakyReLU(alpha=0.01), name='stock_gru_output')(x1)

    # Combine processed inputs
    combined = Concatenate(name='concat_output')([x0_option_dense_output, x1_stock_gru_output])
    
    combined = Dense(16, activation="relu")(combined)
    outputs = Dense(2, activation="linear")(combined)

    model = Model(inputs=[inputOptionData, inputStockData], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

    return model


# test Ryan's model
def make_model(option_data_shape, stock_data_shape):
    
    #inputCName = keras.Input(shape=(1), dtype=tf.string)
    #inputOptionData = keras.Input(shape=(NOPTION_DATA, 1))
    #inputStockData = keras.Input(shape=(NSTOCK_DAYS, 1))
    
    # Define inputs for each type of data
    inputOptionData = Input(shape=option_data_shape, name='x_option_train_expanded')
    inputStockData = Input(shape=stock_data_shape, name='x_stock_train')
    
    # X0 (the contract name, a string)
  
    
    # X1
    #Flatten the option data
    x1 = layers.Flatten()(inputOptionData)
    x1 = layers.Dense(32, activation="relu")(x1)
    x1 = layers.Dense(8, activation="relu",name='option_dense_output')(x1)
    #x1 = layers.Dropout(0.1)(x1)
        
    # X2
    #GRU on the stock price data
    x2 = layers.GRU(units = 32, return_sequences = True, activation='tanh')(inputStockData)
    x2 = layers.Dropout(0.2)(x2)
    
    x2 = layers.GRU(units = 32, return_sequences = True, activation='tanh')(x2)
    x2 = layers.Dropout(0.2)(x2)    

    x2 = layers.GRU(units = 32, return_sequences = True, activation='tanh')(x2)
    x2 = layers.Dropout(0.2)(x2) 
    
    x2 = layers.GRU(units = 32, return_sequences = True, activation='tanh')(x2)
    x2 = layers.Dropout(0.2)(x2) 
    
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(32, activation="relu")(x2)
    x2 = layers.Dense(8, activation="relu",name='stock_gru_output')(x2)
    #x2 = layers.Dropout(0.1)(x2)
    
    
    # X3
    #Convo layer on stock price data
    x3 = layers.Conv1D(32, kernel_size = 1, strides=1, activation="relu")(inputStockData)
    x3 = layers.Conv1D(32, kernel_size = 3, strides=1, activation="relu")(x3)
    x3 = layers.Conv1D(32, kernel_size = 5, strides=1, activation="relu")(x3)
    x3 = layers.MaxPooling1D(2)(x3)
    x3 = layers.Conv1D(32, kernel_size = 3, strides=1, activation="relu")(x3)
    x3 = layers.MaxPooling1D(2)(x3)
    x3 = layers.Flatten()(x3)
    x3 = layers.Dense(32, activation="relu")(x3)
    x3 = layers.Dense(8, activation="relu",name='stock_convo_output')(x3)
    #x3 = layers.Dropout(0.1)(x3)
    
    # X4
    # Dense layer on the stock data
    x4 = layers.Flatten()(inputStockData)
    x4 = layers.Dense(32, activation="relu")(x4)
    x4 = layers.Dense(8, activation="relu",name='stock_dense_output')(x4)
    
    #Combine the option and stock data
    x = layers.Concatenate(name='concat_output')([x1, x3, x4])

    x = layers.Dense(16, activation="relu")(x)
    #x = layers.Dropout(0.1)(x) 
    x = layers.Dense(2)(x)
    
    outputs = x
    
    model = keras.Model([inputOptionData,inputStockData], outputs, name='ryan_GRU_model')
    #model = keras.Model(inputOptionData, outputs, name=name)
    model.summary()   
    
    return model
