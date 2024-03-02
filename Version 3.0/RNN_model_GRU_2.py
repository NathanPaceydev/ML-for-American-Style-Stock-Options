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




def build_gru_model_2(string_lookup_layer, option_data_shape, stock_data_shape):
    
    # Assuming `x_contract_names` is your list or array of contract names.
    string_lookup = StringLookup(output_mode="multi_hot")
    # Adapt with actual contract names data
    string_lookup.adapt(x_contract_names)
    
    # Define inputs for each type of data
    contract_name_input = Input(shape=(None,), dtype='int32', name='encoded_contract_name')
    inputOptionData = Input(shape=option_data_shape, name='option_data')
    inputStockData = Input(shape=stock_data_shape, name='stock_data')

    # Contract name processing
    x0 = string_lookup(contract_name_input)
    x0 = Dense(16, activation="relu")(x0)
    x0_embedding_output = Dense(4, activation="relu", name='embedding_output')(x0)

    # Option data processing with normalization
    x1_norm = LayerNormalization()(inputOptionData)
    x1 = layers.Flatten()(x1_norm)
    x1 = Dense(32, activation="relu")(x1)
    x1_option_dense_output = Dense(8, activation="relu", name='option_dense_output')(x1)

    # Stock data processing with GRU and normalization
    x2_norm = LayerNormalization()(inputStockData)
    x2 = GRU(32, return_sequences=True, activation='tanh')(x2_norm)
    x2 = Dropout(0.2)(x2)
    x2 = layers.Flatten()(x2)
    x2_stock_gru_output = Dense(8, activation="relu", name='stock_gru_output')(x2)

    # Combine processed inputs
    combined = Concatenate(name='concat_output')([x0_embedding_output, x1_option_dense_output, x2_stock_gru_output])
    combined = Dense(16, activation="relu")(combined)
    outputs = Dense(2, activation="linear")(combined)

    model = Model(inputs=[inputCName, inputOptionData, inputStockData], outputs=outputs, name='GRU_option_model')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model
