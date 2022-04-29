import os
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

from src.models.walk_forward_predictor import WalkForwardPredictor
from src.models.mlp import MultiLayerPerceptron

from src.utils import series_to_supervised

# TODO: Add description! Mention datasources

# Get data path or create a directory if it does not exist
# TODO: This is hacky. Need to fix
pathlib.Path(os.path.join(os.path.dirname(os.getcwd()), "..", "data")).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(os.path.dirname(os.getcwd()), "..", "data")

# Check if file exists
if not os.path.exists(os.path.join(data_path, "s&p500_index.csv")):
    raise ValueError("No data in data folder!")

# Get index data
index_data = pd.read_csv(os.path.join(data_path, "s&p500_index.csv"), index_col=0)
index_data = index_data.to_frame().ffill().dropna()
dates = index_data

n_features = 2

index_data = series_to_supervised(index_data, n_in=n_features, n_out=1)
input_data = index_data.drop(['var1(t)'], axis=1)
output_data = index_data.drop(['var1(t-2)', 'var1(t-1)'], axis=1)


# Create MLP model
mlp_model = MultiLayerPerceptron(
    name="mlp_index_wf",
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
wf_model = WalkForwardPredictor(model=mlp_model, start_date="2004-11-08", end_date="2021-06-01",
                                input_pct_change=1, output_pct_change=1, window_size=252, frequency=7,
                                prediction_length=10, validation_size=21, sliding_window=True,
                                random_validation=False, train_from_scratch=False)

# Train our model through time, and obtain the predictions and errors
mlp_predictions, mlp_error = wf_model.train_and_predict(input_data, output_data)

print("MLP Walk Forward")

print(mlp_predictions)
print(mlp_error)

# sav_dates = pd.DataFrame(mlp_error)
# sav_dates = sav_dates.reset_index()
#
# saved = pd.read_csv(r'C:/Users/ELNA SIMONIS/Documents/Results/TESTING.csv')
# saved = saved.drop(['Unnamed: 0'], axis=1)
#
# saved['Dates'] = sav_dates['Date']
# saved = saved.set_index('Dates')
# saved['error'] = saved['TRUE'] - saved['PRED']
# saved = saved.dropna()
#
# # Calculate RMSE
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from math import sqrt
#
# mse = mean_squared_error(saved['TRUE'], saved['PRED'])
# rmse = sqrt(mean_squared_error(saved['TRUE'], saved['PRED']))
# mae = mean_absolute_error(saved['TRUE'], saved['PRED'])
#
# # Create a plot of our errors through time
#
# plt.figure(figsize=(10, 5))
# figuur = saved['error'] ** 2.0
# figuur.plot(color='blue')
# plt.xlabel('Dates', fontsize=15, fontweight='bold', color='black')
# plt.ylabel('Error', fontsize=15, fontweight='bold', color='black')
# plt.yticks(fontsize=10)
# plt.xticks(fontsize=10)
# plt.show()
# plt.close()
