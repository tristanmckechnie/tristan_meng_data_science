import warnings
# This is hacky, but due to tensorflow requiring lower numpy version, but pmdarima requiring higher version, this is
# done to clear the console.
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from Judene_Code.lstm import LSTMNetwork
from Judene_Code.utils import series_to_supervised


# =====================================================================================================================
# GET DATA
# =====================================================================================================================

# Set number of features - this is the lag length
n_features = 2

# Get index data
index_data = pd.read_csv('./Judene_Code/s&p500/s&p500_index.csv',index_col=0)#pd.read_csv("./test_data/S&P500_yfinance.csv")
index_data = index_data.ffill().dropna()
# index_data = index_data[['Date','Close']].reset_index(drop=True).set_index('Date')

# Make data stationary
index_data = index_data.pct_change()

print(index_data.info())

# Create supervised learning problem
index_data = series_to_supervised(index_data, n_in=n_features, n_out=1)
index_data = index_data.fillna(0.0)

# Create training and testing data
x_train, x_test, y_train, y_test = train_test_split(index_data.iloc[:, :-1], index_data.iloc[:, -1],
                                                    test_size=0.1, random_state=1, shuffle=False)

# Create validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.01, random_state=1, shuffle=False)

print("x_train: ", np.shape(x_train))
print("x_val: ", np.shape(x_val))
print("y_train: ", np.shape(y_train))
print("y_val: ", np.shape(y_val))

# =====================================================================================================================
# BUILD MODEL
# =====================================================================================================================


# Create LSTM
lstm = LSTMNetwork(
    name="lstm_index_nwf",
    num_inputs=n_features,
    num_outputs=1,
    # If true, training info is outputted to stdout
    keras_verbose=True,
    # A summary of the NN is printed to stdout
    print_model_summary=True,
    # lstm_layers = [units, kernel_regularizer (l2), recurrent_regularizer (l2), dropout, recurrent_dropout]
    lstm_layers=[
        [50, 0.0, 0.0, 0.2, 0.0],
        [50, 0.0, 0.0, 0.2, 0.0],
        [50, 0.0, 0.0, 0.2, 0.0]
    ],
    # Statefulness
    stateful_training=False,
    # ff_layers = [units, activation, regularization, dropout, use_bias]
    ff_layers=[
        [512, "relu", 0.0, 0.2, True, "gaussian"],
        [512, "relu", 0.0, 0.2, True, "gaussian"],
        [512, "relu", 0.0, 0.2, True, "gaussian"]
    ],
    # The final output layer's activation function
    final_activation="tanh",
    # The objective function for the NN
    objective="mse",
    # The maximum number of epochs to run
    epochs=2000,
    # The batch size to use in the NN
    batch_size=32,
    # The learning rate used in optimization
    learning_rate=0.001,
    # If this many stagnant epochs are seen, stop training
    stopping_patience=15
)

# Train MLP model from scratch
lstm.train(x_train, y_train, x_val, y_val, load_model=False)

# Predict
predicted = lstm.predict(x_test)

# Create dataframe for predicted values
pred_df = pd.DataFrame(np.column_stack([np.squeeze(predicted), y_test]))
pred_df.columns = ["PRED", "TRUE"]

# Plot predicted values
pred_df.plot()
plt.show()
plt.close()
