"""
Walk forward validation methods. This module contains methods for implementing walkforward
validation, with or without feature engineering applied.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
from one_dimensional_time_series_forecasting import time_series_prediction, hit_rate


# transform univariate time series to supervised learning problem
def series_to_supervised(time_series,lag_window_length):
        # initialize input array
        num_rows = len(time_series) - lag_window_length
        array = np.zeros((num_rows, lag_window_length + 1))
        
        # loop through data and populate array
        for i in range(num_rows):
            # input features
            array[i,0:lag_window_length+1] = time_series[i:i+lag_window_length+1]
            # target feature/s
            array[i,-1] = time_series[i+lag_window_length]


        # save results as a class attribute
        input_data = array[:,0:lag_window_length]
        target_data = array[:,lag_window_length]

        return input_data, target_data

def walk_forward_val(model_name,model,original_series,time_series_dates,lag_window_length,train_len=220,test_len=30,train_frequency=5, transformer=None,only_training=True, **kwargs):
    """
    This method implements a walk forward validation technique.
    param: model:     trained time series model  
    param: original_series: the original time series before soos feature engineered the training set portion.
    param: train_len: n number of training samples [0:n)
    param: test_len:  m number of testing samples [n:m]
    param: train_frequency: retrain model every f windows
    param: transformer: this is a function used to feature engineer training data during walk forward validation
    param: only_training: only feature engineer the training set during walk forward validation. Used for denosing, when we dont denoise the testing data.
    param: feat_type: this is a string used to define what feature engineering will be happening.

    Possible kwargs depending on type of feature engineering employed, these kwargs are passed to the feature engineering function - transformer:
        - verbose=False : for plotting stuff
        - wavelet='sym8' : which wavelet to use for denoising
        - threshold_overide: used when the users wants to set a predetermined threshold for denosing
        - threshold: the threshold value the user wants to implement.
    """
    
    # variables to hold results through time
    predictions = list() # list of predictions through time
    history = list()     # list of real values through time  
    history_alt = list() # using different values tro track real through time 

    """
    importantly: these values throught time will run for dates: 0+lag_length --> -1 (ie last)
    """ 

    # define walk forward parameters
    step_size = test_len                 # how many time steps forward the validation takes, = test_len ensures all timesteps are tested
    window_length = train_len + test_len # how many time steps in a single windows test-train dataset 
    # plt.figure()
    # plt.plot(original_series)
    # how many walk forward validation steps to take
    num_walks = int((len(original_series)-train_len) / step_size)
    print(f'Taking {num_walks} walks during walk forward validation')

    # only perform walk forward val if there are a whole number of walks
    if len(original_series) % step_size == 0:

        # loop through the forward walks
        for walk in range(num_walks):
            print(f'walk {walk}')
            df_walk = pd.DataFrame(columns=['original','feature_engineered'])
            # define walk start and end point in time
            walk_start = walk*step_size
            walk_end = walk*step_size + window_length

            # subset data to a single walk
            walk_dataset = original_series[walk_start:walk_end]

            # apply some feature engineering during walk forward validation
            if transformer is not None:
                if only_training == True:
                    # feature engineer only training dataset
                    feature_engineered_training = transformer(walk_dataset[0:train_len],**kwargs)#threshold_override=False,threshold=0.5,verbose=verbose)
                    normal_testing = original_series[walk_start+train_len:walk_end]
                    # add the unfeature engineered tail
                    walk_dataset = np.append(feature_engineered_training,normal_testing)
                    
                else:
                    # feature engineer entire dataset
                    walk_dataset = transformer(walk_dataset,**kwargs)#threshold_override=False,threshold=0.5,verbose=verbose)

            # series to supervised ml task, training data is a matrix, testing data is a 1-d array for a univariate time series forecasting problem
            training_data, testing_data = series_to_supervised(walk_dataset,lag_window_length)

            # test train split
            X_train = training_data[0:-test_len,:]
            X_test =  training_data[-test_len:,:]

            y_train = testing_data[0:-test_len]
            y_test =  testing_data[-test_len:]

            # retrain model
            if walk % train_frequency == 0:
                print('Retraining model.')
                if model_name != 'LSTM':
                    model = model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

            # make predictions
            walk_preds = model.predict(X_test)
            
            # store results
            predictions.extend(walk_preds)
            history.extend(y_test)
            
        print(f'len pred: ', len(predictions))
        print(f'len history: ', len(history))
        print(f'len dates: ', len(time_series_dates[train_len:]))

        # now determine average evaluation metrics through time
        mse = mean_squared_error(history,predictions)
        mae = mean_absolute_error(history,predictions)
        mape = mean_absolute_percentage_error(history,predictions)
        df, accuracy = hit_rate(time_series_dates[train_len:],history,predictions)

        print('MAPE:',mape)
        print('RMSE: ',np.sqrt(mse))
        print('MAE: ',mae)
        print('Directional Accuracy: ', accuracy)

        # store everything into a dataframe
        df_walk_forward = pd.DataFrame(columns=['date','real_value','prediction'])
        df_walk_forward['date'] = time_series_dates[train_len:]
        df_walk_forward['real_value'] = history
        df_walk_forward['prediction'] = predictions

        return df_walk_forward, df, mse,mae,mape,accuracy

    else:
        print('Not a whole number of walks!')
