"""
This script contains a Tensorflow generalised implementation of an LSTM Model.
"""

from math import sqrt
from numpy import array
from numpy import mean
from numpy import std

from pandas import DataFrame
from pandas import concat
from pandas import read_csv

from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot

# fit a model
def lstm(n_input, n_nodes, n_lstm_layers, n_epochs, n_batch, n_diff, train_x, train_y,verbose=1):

    """
    This function trains a keras lstm neural network. Some important parameters are:

    :param n_input: the lag number
    :param n_nodes: number of hidden units - list with number of nodes per lstm layer
    :param n_lstm_layers: number of lstm hidden layers
    :param n_epochs: number of times training passes over the entire training set
    :param n_batch: how many training samples before weights are updated
    """

    # define model
    model = Sequential()
    
    # add initial lstm layers
    model.add(LSTM(n_nodes[0], activation= ' relu ' , input_shape=(n_input, 1))) # lstm layer

    # add subsequent hidden LSTM layers
    if n_lstm_layers > 1:
        for i in range(1,n_lstm_layers):
            model.add(LSTM(n_nodes[i], activation= 'relu'), return_sequences=True)

    # add a final fully connected layer + single predictor layer 
    model.add(Dense(1))                                                       # final predictor

    # prepare optimizer and compile model
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss= 'mse', optimizer= opt)

    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)

    # show model summary
    model.summary()

    return model

# forecast with a pre-fit model
def model_predict(model, history, config):
    # unpack config
    n_input, _, _, _, n_diff = config
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # fit model
    model = model_fit(train, cfg)

    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = model_predict(model, history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    print( ' > %.3f ' % error)
    return error