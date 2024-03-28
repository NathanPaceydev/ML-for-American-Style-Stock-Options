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
    x1 = Bidirectional(GRU(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001)))(inputStockData)
    
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



def new_GRU_Hybrid_model(option_data_shape, stock_data_shape):
    # Define inputs
    inputOptionData = Input(shape=option_data_shape, name='option_input')
    inputStockData = Input(shape=stock_data_shape, name='stock_input')
    
    # Option data processing
    option_layer = layers.Flatten()(inputOptionData)
    option_layer = layers.Dense(32, activation="relu")(option_layer)
    option_layer = layers.Dense(8, activation="relu", name='option_dense_output')(option_layer)
        
    # GRU layers on stock data
    stock_gru_layer = layers.GRU(32, return_sequences=True, activation='tanh')(inputStockData)
    stock_gru_layer = layers.Dropout(0.2)(stock_gru_layer)
    stock_gru_layer = layers.GRU(32, return_sequences=True, activation='tanh')(stock_gru_layer)
    stock_gru_layer = layers.Dropout(0.2)(stock_gru_layer)
    stock_gru_layer = layers.Flatten()(stock_gru_layer)
    stock_gru_layer = layers.Dense(32, activation="relu")(stock_gru_layer)
    stock_gru_layer = layers.Dense(8, activation="relu", name='stock_gru_output')(stock_gru_layer)
    
    # Convolutional layers on stock data
    stock_conv_layer = layers.Conv1D(32, kernel_size=1, activation="relu")(inputStockData)
    stock_conv_layer = layers.Conv1D(32, kernel_size=3, activation="relu")(stock_conv_layer)
    stock_conv_layer = layers.Conv1D(32, kernel_size=5, activation="relu")(stock_conv_layer)
    stock_conv_layer = layers.MaxPooling1D(2)(stock_conv_layer)
    stock_conv_layer = layers.Conv1D(32, kernel_size=3, activation="relu")(stock_conv_layer)
    stock_conv_layer = layers.MaxPooling1D(2)(stock_conv_layer)
    stock_conv_layer = layers.Flatten()(stock_conv_layer)
    stock_conv_layer = layers.Dense(32, activation="relu")(stock_conv_layer)
    stock_conv_layer = layers.Dense(8, activation="relu", name='stock_convo_output')(stock_conv_layer)
    
    # Dense layer on stock data
    stock_dense_layer = layers.Flatten()(inputStockData)
    stock_dense_layer = layers.Dense(32, activation="relu")(stock_dense_layer)
    stock_dense_layer = layers.Dense(8, activation="relu", name='stock_dense_output')(stock_dense_layer)
    
    # Combine layers
    combined_layer = layers.Concatenate(name='concat_output')([option_layer, stock_conv_layer, stock_dense_layer])
    combined_layer = layers.Dense(16, activation="relu")(combined_layer)
    outputs = layers.Dense(2, activation="linear")(combined_layer)
    
    # Construct the model
    model = keras.Model(inputs=[inputOptionData, inputStockData], outputs=outputs, name='New_Hybrid_GRU_Model')
    model.summary()

    return model

