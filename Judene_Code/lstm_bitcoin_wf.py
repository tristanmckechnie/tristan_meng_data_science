import os
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

from src.models.walk_forward_predictor import WalkForwardPredictor
from src.models.lstm import LSTM

from src.utils import series_to_supervised

# TODO: Add description! Mention datasources
# TODO: Create a "visualizer" class that can be repeatedly used to create graphs

# Get data path or create a directory if it does not exist
# TODO: This is hacky. Need to fix
pathlib.Path(os.path.join(os.path.dirname(os.getcwd()), "..", "data")).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(os.path.dirname(os.getcwd()), "..", "data")

# Check if file exists
if not os.path.exists(os.path.join(data_path, "local_etfs_close.csv")):
    raise ValueError("No data in data folder!")

# Get bitcoin data
bitcoin_data = pd.read_csv(os.path.join(data_path, "bitcoin.csv"), index_col=0)
bitcoin_data = bitcoin_data.to_frame().ffill().dropna()
dates = bitcoin_data

n_features = 2

bitcoin_data = series_to_supervised(bitcoin_data, n_in=n_features, n_out=1)
input_data = bitcoin_data.drop(['var1(t)'], axis=1)
output_data = bitcoin_data.drop(['var1(t-2)', 'var1(t-1)'], axis=1)




# Create LSTM model
lstm_model = LSTM(
    name="lstm_bitcoin_wf",
    num_inputs=n_features,
    num_outputs=1,
    # If true, training info is outputted to stdout
    keras_verbose=False,
    # A summary of the NN is printed to stdout
    print_model_summary=True,
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
    epochs=5,
    # The batch size to use in the NN
    batch_size=64,
    # The learning rate used in optimization
    learning_rate=0.001,
    # If this many stagnant epochs are seen, stop training
    stopping_patience=50
)

# Initiate our model
wf_model = WalkForwardPredictor(model=gru_model, start_date="2004-11-08", end_date="2021-06-01",
                                input_pct_change=1, output_pct_change=1, window_size=252, frequency=7,
                                prediction_length=10, validation_size=21, sliding_window=True,
                                random_validation=False, train_from_scratch=False)

# Train our model through time, and obtain the predictions and errors
lstm_predictions, lstm_error = wf_model.train_and_predict(input_data, output_data)

print("LSTM Walk Forward")

print(lstm_predictions)
print(lstm_error)
