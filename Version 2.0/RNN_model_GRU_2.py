import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
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