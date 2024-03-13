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
    inputOptionData = Input(shape=option_data_shape, name='option_data')
    inputStockData = Input(shape=stock_data_shape, name='stock_data')

    # Option data processing with normalization
    x0_norm = LayerNormalization()(inputOptionData)
    x0 = layers.Flatten()(x0_norm)
    x0 = Dense(32, activation="relu")(x0)
    x0_option_dense_output = Dense(8, activation="relu", name='option_dense_output')(x0)

    # Stock data processing with GRU and normalization
    x1_norm = LayerNormalization()(inputStockData)
    x1 = Bidirectional(GRU(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)))(x1_norm)
    x1 = Dropout(0.3)(x1)  # Increased dropout
    x1 = layers.Flatten()(x1)
    x1_stock_gru_output = Dense(8, activation=LeakyReLU(alpha=0.01), name='stock_gru_output')(x1)

    # Combine processed inputs
    combined = Concatenate(name='concat_output')([x0_option_dense_output, x1_stock_gru_output])
    combined = Dense(16, activation="relu")(combined)
    outputs = Dense(2, activation="linear")(combined)

    model = Model(inputs=[inputOptionData, inputStockData], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model
