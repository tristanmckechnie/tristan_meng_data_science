# plotting and data wrangling
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# model evalution metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error

# data preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# predictive models
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# cross validation and hyper-parameter search
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

# neural networks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# class for one-dimensional time series forecasting
class time_series_prediction():

    def __init__(self,dates,one_d_time_series,lag_window_length,n_ahead_prediction):

        # raw input data + settings for time series -> supervised learning ML problem
        self.one_d_time_series = one_d_time_series #np.array(one_d_time_series)      # time series array, to array ensure index works as expected for class methods
        self.time_series_dates = np.array(dates)                  # time stamp / date for each data point
        self.lag_window_length = lag_window_length                # length of lag window
        self.n_ahead_prediction = n_ahead_prediction              # time ahead to predict

        # transfromed data: set after calling .sliding_window_1()
        self.input_data = None
        self.target_data = None

        # testing and training data: set after calling .train_test_split()
        self.training_split = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None

        # predictions from various models - set after calling each models training
        self.linear_reg_predictions = None
        self.svm_predictions = None
        self.neural_net_predictions = None
        self.naive_predictions = None
        self.lstm_predictions = None

        # cumprod results from predictions - set after calling .vis_results_time_series()
        self.real_vals_cumprod = None
        self.linear_reg_predictions_cumprod = None
        self.svm_predictions_cumprod = None
        self.neural_net_predictions_cumprod = None

        # model hyperparameter grid search results
        self.nn_grid_params = None

        # model testing metric results
        self.linear_reg_rmse = None
        self.svm_rmse = None
        self.nn_rmse = None
        self.naive_rmse = None
        self.lstm_rmse = None

        # model loss-curves
        self.nn_loss_curve = None 
        self.svm_lost_curve = None
        self.lstm_loss_curve = None

        # models
        self.linear_regression_model = None
        self.svr_model = None
        self.mlp_model = None
        self.lstm_model = None

# ****************************************************************************************************************
    # data wrangling
# ****************************************************************************************************************

    # method to transfroms 1-D time series to supervised ML problem: one step ahead forecasting   
    def sliding_window_1(self,verbose):
        # initialize input array
        num_rows = len(self.one_d_time_series) - self.lag_window_length
        array = np.zeros((num_rows, self.lag_window_length + 1))
        
        # loop through data and populate array
        for i in range(num_rows):
            # input features
            array[i,0:self.lag_window_length+1] = self.one_d_time_series[i:i+self.lag_window_length+1]
            # target feature/s
            array[i,-1] = self.one_d_time_series[i+self.lag_window_length]
            
            if verbose == 1:
                # show pattern
                print(array[i,0:self.lag_window_length],' : ',array[i,self.lag_window_length])

        # save results as a class attribute
        self.input_data = array[:,0:self.lag_window_length]
        self.target_data = array[:,self.lag_window_length]

    # method to perform a training and testing split for dataset with only a single column of target variables
    def train_test_split(self,split):
        # sequentially splits data for testing and training
        self.training_split = split
        self.X_train = self.input_data[0:-split,:]
        self.X_test = self.input_data[-split:,:]
        self.y_train = self.target_data[0:-split]
        self.y_test = self.target_data[-split:]

        # generate different folds from training data for cross validation during hyperparameter tuning

        # different folds for cross validation
        tscv = TimeSeriesSplit(n_splits=5)

        # visualize cross validation splits
        fig,ax = plt.subplots(5,1,sharex=True)
        i = 0
        training_data = self.one_d_time_series[0:-self.training_split]
        for tr_index, val_index in tscv.split(training_data): # training and validation splits for 5 folds
            # print(tr_index, val_index)
            ax[i].plot(tr_index,training_data[tr_index[0]:tr_index[-1]+1],'b-',label='training set')
            ax[i].plot(val_index,training_data[val_index[0]:val_index[-1]+1],'r-',label='validation set')
            ax[i].legend()
            i += 1
        ax[0].set_title('Cross validation sets for hyperparameter tuning')
        plt.tight_layout()
        plt.show()

    
    # this method prepares a supervised ml dataset from a time series and is used in the walk forward validation method
    def series_to_supervised(self,time_series):
        # initialize input array
        num_rows = len(time_series) - self.lag_window_length
        array = np.zeros((num_rows, self.lag_window_length + 1))
        
        # loop through data and populate array
        for i in range(num_rows):
            # input features
            array[i,0:self.lag_window_length+1] = time_series[i:i+self.lag_window_length+1]
            # target feature/s
            array[i,-1] = time_series[i+self.lag_window_length]


        # save results as a class attribute
        input_data = array[:,0:self.lag_window_length]
        target_data = array[:,self.lag_window_length]

        return input_data, target_data
    
    # this method implements walk foward validation training and testing
    def walk_forward_val(self,model_name,model,train_len=220,test_len=30,train_frequency=5):
        """
        This method implements a walk forward validation technique.
        param: model:     trained time series model  
        param: train_len: n number of training samples [0:n)
        param: test_len:  m number of testing samples [n:m]
        param: train_frequency: retrain model every f windows
        """
        
        # variables to hold results through time
        predictions = list() # list of predictions through time
        history = list()     # list of real values through time   

        """
        importantly: these values throught time will run for dates: 0+lag_length --> -1 (ie last)
        """ 

        # define walk forward parameters
        step_size = test_len                 # how many time steps forward the validation takes, = test_len ensures all timesteps are tested
        window_length = train_len + test_len # how many time steps in a single windows test-train dataset 

        # how many walk forward validation steps to take
        num_walks = int((len(self.one_d_time_series)-train_len) / step_size)
        print(f'Taking {num_walks} walks during walk forward validation')

        # only perform walk forward val if there are a whole number of walks
        if len(self.one_d_time_series) % step_size == 0:

            # loop through the forward walks
            for walk in range(num_walks):
                print(f'walk {walk}')
                # define walk start and end point in time
                walk_start = walk*step_size
                walk_end = walk*step_size + window_length

                # subset data to a single walk
                walk_dataset = self.one_d_time_series[walk_start:walk_end]

                # series to supervised ml task
                training_data, testing_data = self.series_to_supervised(walk_dataset)

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
            print(f'len dates: ', len(self.time_series_dates[train_len:]))

            # now determine average evaluation metrics through time
            mse = mean_squared_error(history,predictions)
            mae = mean_absolute_error(history,predictions)
            mape = mean_absolute_percentage_error(history,predictions)
            df, accuracy = hit_rate(self.time_series_dates[train_len:],history,predictions)

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)
            print('Directional Accuracy: ', accuracy)

            # store everything into a dataframe
            df_walk_forward = pd.DataFrame(columns=['date','real_value','prediction'])
            df_walk_forward['date'] = self.time_series_dates[train_len:]
            df_walk_forward['real_value'] = history
            df_walk_forward['prediction'] = predictions

            return df_walk_forward, df

        else:
            print('Not a whole number of walks!') 
                         


# ****************************************************************************************************************
    # predictive models
# ****************************************************************************************************************

    def linear_regression(self):
        print('Training multivariate linear regression:')
        # train model
        reg_model = LinearRegression().fit(self.X_train,self.y_train)
        print('\nLinear regression coefficients: \n',reg_model.coef_)

        # test model
        predictions = reg_model.predict(self.X_test)

        # evaluate: use sklearn metric methods to calc rmse and mae
        mse = mean_squared_error(self.y_test,predictions)
        mae = mean_absolute_error(self.y_test,predictions)
        mape = mean_absolute_percentage_error(self.y_test,predictions)

        print('MAPE:',mape)
        print('RMSE: ',np.sqrt(mse))
        print('MAE: ',mae)

        # save predictions and results
        self.linear_reg_predictions = predictions
        self.linear_reg_rmse = np.sqrt(mse)

        # save model as class variables
        self.linear_regression_model = reg_model


    def support_vector_machine(self,model_tunning=True,C=None,kernel=None,epsilon=None,verbose=0):
        print('\nTraining support vector machine:')

        if model_tunning == False: #hyperparameter are known
            # train model
            svm_regres = SVR(max_iter=5000,C=C, kernel=kernel, epsilon=epsilon).fit(self.X_train,self.y_train)
            print('Model params: ', svm_regres.get_params())
            # predict on test set
            svm_predictions = svm_regres.predict(self.X_test)

            # evaluate
            mse = mean_squared_error(self.y_test,svm_predictions[:])
            mae = mean_absolute_error(self.y_test,svm_predictions[:])
            mape = mean_absolute_percentage_error(self.y_test,svm_predictions[:])

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)
            
            # save predictions and results
            self.svm_predictions = svm_predictions
            self.svm_rmse = np.sqrt(mse)

            # save model as class variable
            self.svr_model = svm_regres

        else: # must hyperparameter tune model

            # define model: support vector machine for regression
            model = SVR(max_iter=5000,tol=1e-5)

            # hyperparameter values to check
            param_grid = [
            {'C': [0.001,0.1, 1, 10, 100], 'kernel': ['linear','rbf','sigmoid']}, # ,'epsilon':[0.1,1,10,100]
            ]

            # perform grid search, using cross validaiton
            tscv = TimeSeriesSplit(n_splits=5)
            gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_grid, scoring = 'neg_root_mean_squared_error',verbose=verbose,n_jobs=-4)
            gsearch.fit(self.X_train, self.y_train)
            print('best_score: ', gsearch.best_score_)
            print('best_model: ', gsearch.best_estimator_)
            print('best_params: ',gsearch.best_params_)

            # predict on test set
            svm_predictions = gsearch.best_estimator_.predict(self.X_test)

            # evaluate
            mse = mean_squared_error(self.y_test,svm_predictions[:])
            mae = mean_absolute_error(self.y_test,svm_predictions[:])
            mape = mean_absolute_percentage_error(self.y_test,svm_predictions[:])

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)

            # save predictions and results
            self.svm_predictions = svm_predictions
            self.svm_rmse = np.sqrt(mse)

            # save model as class variable
            self.svr_model = gsearch.best_estimator_

    def neural_net_mlp(self,verbose=0,model_tunning=True,hidden_layer_sizes=None,activation=None,learning_rate=None,learning_rate_init=None,solver='adam'):
        print('\nTraining MLP neural network: ')

        if model_tunning == False:
            # train neural network
            nn_regres = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                    activation=activation,
                                    learning_rate=learning_rate,
                                    learning_rate_init=learning_rate_init,
                                    shuffle=False,random_state=1,
                                    max_iter=2000,verbose=verbose,
                                    n_iter_no_change=100,
                                    solver=solver
                                    ).fit(self.X_train,self.y_train)
            print('Model params:', nn_regres.get_params())
            # make predictions
            nn_predictions = nn_regres.predict(self.X_test)

            # evaluate
            mse = mean_squared_error(self.y_test,nn_predictions[:])
            mae = mean_absolute_error(self.y_test,nn_predictions[:])
            mape = mean_absolute_percentage_error(self.y_test,nn_predictions[:])

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)

            # save predictions
            self.neural_net_predictions = nn_predictions
            self.nn_rmse = np.sqrt(mse)

            # save loss-curve
            if solver != 'lbfgs':
                self.nn_loss_curve = nn_regres.loss_curve_

            # save model as class attribute
            self.mlp_model = nn_regres
        
        else: # perform hyperparameter tuning
            MLP = MLPRegressor(shuffle=False,max_iter=5000,tol=1e-4,n_iter_no_change=200,solver=solver) # must set shuffle to false to avoid leakage of information due to sequance problem

            # hyperparameter values to check
            param_grid = [
            {'hidden_layer_sizes': [(10,),(100,),(500,),(1000,),(10,10,10),(100,100,100)], 'activation': ['logistic', 'tanh', 'relu'],'learning_rate': ['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.0001,0.001,0.01]}
 ]
            # perform grid search, using cross validaiton
            tscv = TimeSeriesSplit(n_splits=5)
            gsearch = GridSearchCV(estimator=MLP, cv=tscv, param_grid=param_grid, scoring = 'neg_root_mean_squared_error',verbose=verbose,n_jobs=-4)
            gsearch.fit(self.X_train, self.y_train)
            print('best_score: ', gsearch.best_score_)
            print('best_model: ', gsearch.best_estimator_)
            print('best_params: ',gsearch.best_params_)

            # save grid search parameters
            self.nn_grid_params = pd.DataFrame.from_dict(gsearch.cv_results_)

            # model
            mlp_predictions = gsearch.best_estimator_.predict(self.X_test)

            # evaluate
            mse = mean_squared_error(self.y_test,mlp_predictions)
            mae = mean_absolute_error(self.y_test,mlp_predictions)
            mape = mean_absolute_percentage_error(self.y_test,mlp_predictions)

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)

             # save predictions
            self.neural_net_predictions = mlp_predictions
            self.nn_rmse = np.sqrt(mse)

            # save loss-curve
            self.nn_loss_curve = gsearch.best_estimator_.loss_curve_

            # save model as class attribute
            self.mlp_model = gsearch.best_estimator_

    def lstm(self,model_tunning=True, n_nodes=[128,128,128], n_lstm_layers=3, n_epochs=2000, n_batch=128,verbose=1):
        """
        This function trains a keras lstm neural network. Some important parameters are:

        :param model_tunning:   whether to perform hyperparameter optimization or now
        :param n_nodes:         number of hidden units - list with number of nodes per lstm layer
        :param n_lstm_layers:   number of lstm hidden layers
        :param n_epochs:        number of times training passes over the entire training set
        :param n_batch:         how many training samples before weights are updated
        """

        print('\nTraining LSTM neural network: ')

        if model_tunning == False:
            # define model
            model = Sequential()
            
            # add initial lstm layers
            model.add(LSTM(n_nodes[0], activation= 'relu' , input_shape=(self.lag_window_length, 1),return_sequences=True)) # lstm layer

            # add subsequent hidden LSTM layers
            if n_lstm_layers > 1:
                # setup hidden layers between first and last LSTM layers
                for i in range(1,n_lstm_layers-1):
                    model.add(LSTM(n_nodes[i], activation= 'relu', return_sequences=True))

                # add final lstm hidden layers
                model.add(LSTM(n_nodes[i+1], activation= 'relu', return_sequences=False))

            # add a final fully connected layer + single predictor layer 
            model.add(Dense(1))                                                    # final predictor

            # prepare optimizer and compile model
            opt = Adam(learning_rate=0.001)
            model.compile(loss= 'mse', optimizer= opt)

            # setup some callbacks
            callbacks_list = [EarlyStopping(verbose=1,monitor='loss',mode='min',patience=20)]

            # fit model
            history = model.fit(self.X_train, self.y_train, epochs=n_epochs, batch_size=n_batch, verbose=verbose, callbacks=callbacks_list)
            
            # test set preidictions
            lstm_predictions = model.predict(self.X_test)
            lstm_predictions = np.ravel(lstm_predictions[:,0]) # reshape from (X,1) to (X,)

            # evaluate
            mse = mean_squared_error(self.y_test,lstm_predictions)
            mae = mean_absolute_error(self.y_test,lstm_predictions)
            mape = mean_absolute_percentage_error(self.y_test,lstm_predictions)

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)

             # save predictions
            self.lstm_predictions = lstm_predictions
            self.lstm_rmse = np.sqrt(mse)

            # save loss-curve
            self.lstm_loss_curve = history

            # show model summary
            model.summary()

            # save model as class attribute
            self.lstm_model = model

        elif model_tunning == True:

            # define abstract lstm builder function
            def build_model(hp):

                # define model
                model = Sequential()
                
                # add initial lstm layers
                model.add(LSTM(units=hp.Int("units 0", min_value=16, max_value=288, step=32), activation= 'relu' , input_shape=(self.lag_window_length, 1),return_sequences=True)) # lstm layer

                # setup hidden layers between first and last LSTM layers
                for i in range(hp.Int('layers',1,3)):
                    model.add(LSTM(hp.Int(f"units {i}", min_value=16, max_value=288, step=32), activation= 'relu', return_sequences=True))

                # add final lstm hidden layers
                model.add(LSTM(hp.Int("units -1", min_value=16, max_value=288, step=32), activation= 'relu', return_sequences=False))

                # add a final fully connected layer + single predictor layer 
                model.add(Dense(1))  # final predictor

                # prepare optimizer and compile model
                opt = Adam(learning_rate=hp.Choice('learning_rate',[1e-2,1e-3,1e-4]))
                model.compile(loss= 'mse', optimizer= opt)

                return model

            # setup hyperparameter tuner
            tuner = kt.RandomSearch(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=20,
                executions_per_trial=5,
                overwrite=True,
                directory="./lstm_hp_res/",
                project_name="helloworld",
            )

            # setup data with validation set for hp tuning
            X_train = self.X_train
            Y_train = self.y_train

            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=1, shuffle=False)

            # start tuning hyperparameters
            tuner.search(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val))

            # retrieve and build best LSTM model
            best_hp = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hp)
            history = model.fit(best_hp, model, self.X_train, self.y_train)

            # test set preidictions
            lstm_predictions = model.predict(self.X_test)
            lstm_predictions = np.ravel(lstm_predictions[:,0])

            # evaluate
            mse = mean_squared_error(self.y_test,lstm_predictions)
            mae = mean_absolute_error(self.y_test,lstm_predictions)
            mape = mean_absolute_percentage_error(self.y_test,lstm_predictions)

            print('MAPE:',mape)
            print('RMSE: ',np.sqrt(mse))
            print('MAE: ',mae)

             # save predictions
            self.lstm_predictions = lstm_predictions
            self.lstm_rmse = np.sqrt(mse)

            # save loss-curve
            self.lstm_loss_curve = history

            # save model as class attribute
            self.lstm_model = model


    def naive_model(self): # t's prediction is t-1's value, note that this means you miss the first time point
        preds = np.zeros(self.training_split)
        preds = self.one_d_time_series[-self.training_split-1:-1]

        # evaluate
        mse = mean_squared_error(self.y_test,preds)
        mae = mean_absolute_error(self.y_test,preds)
        mape = mean_absolute_percentage_error(self.y_test,preds)

        print('\nNaive model results:')
        print('MAPE:',mape)
        print('RMSE: ',np.sqrt(mse))
        print('MAE: ',mae)

        # save predictions and results
        self.naive_predictions = preds
        self.naive_rmse = np.sqrt(mse)
        
# ****************************************************************************************************************
    # visualize results
# ****************************************************************************************************************
    def error(self,real_data,predicted_data):
        error = np.zeros(len(real_data))
        error = np.abs((real_data - predicted_data) / real_data)
        return error

    # visualize orignal time series signal aswell as predictions    
    def vis_results_time_series(self,ylabel='value',second_plot='error',steps=150):
        # plot prediction against actual + training data
        fig, ax = plt.subplots(2,1,figsize=(10,8),sharex=True)

        # original time series
        ax[0].plot(self.time_series_dates[-self.training_split:],self.one_d_time_series[-self.training_split:],'-',linewidth=3,label='real values')  # ,markersize=5

        # predicted y values
        if self.linear_reg_predictions is not None:
            ax[0].plot(self.time_series_dates[-self.training_split:],self.linear_reg_predictions,'-',label='linear regression prediction',markersize=5)
        ax[0].plot(self.time_series_dates[-self.training_split:],self.naive_predictions,'-',label='naive prediction')
        if self.svm_predictions is not None:
            ax[0].plot(self.time_series_dates[-self.training_split:],self.svm_predictions,'-',label='SVR prediction')
        if self.neural_net_predictions is not None:
            ax[0].plot(self.time_series_dates[-self.training_split:],self.neural_net_predictions,'-',label='MLP prediction')
        if self.lstm_predictions is not None:
            ax[0].plot(self.time_series_dates[-self.training_split:],self.lstm_predictions,'-',label='LSTM prediction')

        ax[0].legend()
        ax[0].set_title('Real values vs model predictions')
        ax[0].set_ylabel(ylabel)
        

        # plot error plot
        if second_plot == 'error':

            if self.linear_reg_predictions is not None:
                error_linreg = self.error(self.y_test,self.linear_reg_predictions)
            if self.svm_predictions is not None:
                error_svm = self.error(self.y_test,self.svm_predictions)
            if self.neural_net_predictions is not None:
                error_nn = self.error(self.y_test,self.neural_net_predictions)
            if self.lstm_predictions is not None:
                error_lstm = self.error(self.y_test,self.lstm_predictions)

            if self.linear_reg_predictions is not None:
                ax[1].plot(self.time_series_dates[-self.training_split:],error_linreg,'r-',label='linear reg error')#,markerfmt=" ")
            if self.svm_predictions is not None:
                ax[1].plot(self.time_series_dates[-self.training_split:],error_svm,'-',label='SVR error')#,markerfmt=" ")
            if self.neural_net_predictions is not None:
                ax[1].plot(self.time_series_dates[-self.training_split:],error_nn,'-',label='MLP error')#,markerfmt=" ")
            if self.lstm_predictions is not None:
                ax[1].plot(self.time_series_dates[-self.training_split:],error_lstm,'-',label='LSTM error')#,markerfmt=" ")
            
            ax[1].set_title('Error signal for predictive models')
            ax[1].set_xlabel('Dates')
            ax[1].legend()
            # ax[1].set_ylim([-10,10])
            ax[1].set_xticks([self.time_series_dates[x] for x in range(-self.training_split,-1,steps)])
            ax[1].tick_params(rotation=30)
            ax[1].set_ylabel('Error')
            ax[1].set_xlabel('Date')
        
        elif second_plot == 'cumprod':

            # plot cummulative prod plots - this should only be done if input data is percentage retunrs
            self.real_vals_cumprod = (self.y_test+1).cumprod()
            self.linear_reg_predictions_cumprod = (self.linear_reg_predictions + 1).cumprod()
            self.svm_predictions_cumprod = (self.svm_predictions + 1).cumprod()
            self.neural_net_predictions_cumprod = (self.neural_net_predictions + 1).cumprod()

            ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.real_vals_cumprod,'-',label='real vals cumprod')
            ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.linear_reg_predictions_cumprod,'-',label='linear reg cumprod')
            ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.svm_predictions_cumprod,'-',label='svm cumprod')
            ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.neural_net_predictions_cumprod,'-',label='nn cumprod')

            ax[1].set_xticks([self.time_series_dates[x] for x in range(self.training_split,len(self.time_series_dates),28)])
            ax[1].tick_params(rotation=30)
            ax[1].legend()

        # titles and save figures
        # title_string = 'S&P500 predictions _ y is '+str(column)+'_ window len is '+ str(window_length)
        # fig.suptitle(title_string)
        
        # fig_name = '../results/univariate_single_step_ahead/'+title_string+'.png'
        # plt.savefig(fig_name,facecolor='w')
        plt.tight_layout()

    # visualize predictions against real values using scatter plot
    def vis_results_scatter(self):

        # create dataframe to hold all results
        df_predictions = pd.DataFrame(index=self.time_series_dates[self.training_split+self.lag_window_length:],columns=['Real_values','linear_reg_predictions','svm_predictions','neural_net_predictions'])
        df_predictions['Real_values'] = self.y_test
        df_predictions['linear_reg_predictions'] = self.linear_reg_predictions
        df_predictions['svm_predictions'] = self.svm_predictions
        df_predictions['neural_net_predictions'] = self.neural_net_predictions

        # scatter plot with hues
        fig, ax = plt.subplots(3,1,figsize=(7,10))
        sns.scatterplot(y=df_predictions['Real_values'],x=df_predictions['linear_reg_predictions'],ax=ax[0])
        sns.lineplot(x=self.y_test,y=self.y_test,ax=ax[0],color='red')

        sns.scatterplot(y=df_predictions['Real_values'],x=df_predictions['svm_predictions'],ax=ax[1])
        sns.lineplot(x=self.y_test,y=self.y_test,ax=ax[1],color='red')

        sns.scatterplot(y=df_predictions['Real_values'],x=df_predictions['neural_net_predictions'],ax=ax[2])
        sns.lineplot(x=self.y_test,y=self.y_test,ax=ax[2],color='red')

        # plot formatting
        plt.tight_layout()

    # method to plot testing and training split of data
    def test_train_plot(self,ylabel='value',steps=150):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(self.time_series_dates[0:-self.training_split] ,self.one_d_time_series[0:-self.training_split],'ok-',label='Training data',markersize=3) # replace returns with sp_500 for other data plotting
        ax.plot(self.time_series_dates[-self.training_split:] ,self.one_d_time_series[-self.training_split:],'or-',label='Testing data',markersize=3)
        # ax.plot(self.time_series_dates[self.training_split+self.lag_window_length:] ,self.y_test,'o',label='Windowed testing data') # important to match time by start 5 (length of time window) after where segmented our testing and training data
        plt.legend(loc=0) 
        ax.set_xticks([self.time_series_dates[x] for x in range(0,len(self.time_series_dates),steps)])
        ax.tick_params(rotation=30) 
        ax.set_title('Test traing split')
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        plt.tight_layout()

    # method to tabulate all results together nicely
    def results(self):
        """
        This function neatly collects the prediction data into a single dataframe        
        """
        df_results = pd.DataFrame(columns=['date','Value','Linear','SVM','NN','LSTM','Naive'])
        df_results['date'] = self.time_series_dates
        df_results['Value'] = self.one_d_time_series
        
        # set all values before prediction start to zero
        zeros = [None for i in range(len(self.one_d_time_series)-self.training_split)]

        # append prediction results
        if self.linear_reg_predictions is not None:
            linear_predictions = np.append(zeros,self.linear_reg_predictions)
        if self.svm_predictions is not None:
            svm_predictions = np.append(zeros,self.svm_predictions)
        if self.neural_net_predictions is not None:
            nn_predictions = np.append(zeros,self.neural_net_predictions)
        if self.lstm_predictions is not None:
            lstm_predictions = np.append(zeros,self.lstm_predictions)
        naive_predictions = np.append(zeros,self.naive_predictions)

        # save predictions to df
        if self.linear_reg_predictions is not None:
            df_results['Linear'] = linear_predictions
        if self.svm_predictions is not None:
            df_results['SVM'] = svm_predictions
        if self.neural_net_predictions is not None:
            df_results['NN'] = nn_predictions
        if self.lstm_predictions is not None:
            df_results['LSTM'] = lstm_predictions
        df_results['Naive'] = naive_predictions
        
        return df_results



# ****************************************************************************************************************
#  Other useful helper functions
# ****************************************************************************************************************

# this function determines whether predictions determine the correct movement for tomorrow.
def hit_rate(dates,original_values, predictions): # pass lists / arrays of dates, original values, and predictions
    # initialise dataframe
    df = pd.DataFrame(columns=['Date','Original Value','Daily PCT','Movement','Prediction','Predicted Movement'])

    # add known data as passed to function
    if type(dates) != list:
        df['Date'] = list(dates)#.to_list()
    else:
        df['Date'] = dates

    if type(original_values) != list:
            df['Original Value'] = list(original_values)#.to_list()
    else:
            df['Original Value'] = original_values

    if type(predictions) != list:
        df['Prediction'] = list(predictions)#.to_list()
    else:
        df['Prediction'] = predictions

    # determine actually movement from time t to t+1 and predicted movement
    df['Daily PCT'] = df['Original Value'].pct_change() # percentange change between t and t+1
    df['Movement'] = df['Daily PCT'].apply(lambda x: 1 if x > 0 else 0)
    df['Predicted Movement'] = df['Prediction'].pct_change().apply(lambda x: 1 if x > 0 else 0)

    # calculate classification evaluation metrics
    y_true = df['Movement']
    y_pred = df['Predicted Movement']
    matrix = confusion_matrix(y_true,y_pred)
    accuracy = accuracy_score(y_true,y_pred)

    # display eval metrics
    print(f'Movement prediction accuracy: {round(accuracy*100,2)} %')
    print(f'Confusion matrix:\n{matrix}')
    return df, accuracy

def invert_scaling(scaler,testing_data,predictions):
    # invert scaling
    inverted_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    inverted_testing_data = scaler.inverse_transform(testing_data.reshape(-1, 1))
    # compute eval metrics
    mse = mean_squared_error(inverted_testing_data,inverted_predictions)
    mae = mean_absolute_error(inverted_testing_data,inverted_predictions)
    mape = mean_absolute_percentage_error(inverted_testing_data,inverted_predictions)

    print('MAPE:',mape)
    print('RMSE: ',np.sqrt(mse))
    print('MAE: ',mae)

    return inverted_predictions, inverted_testing_data

    # invert difference + log
def invert_first_difference_with_log(prediction_split,lag_window,predictions,df_original):
    # first real value to work from
    beginnning_value = df_original['#Passengers_log'].iloc[-prediction_split] # this must be the column that is logged, before differencing
    beginning_date = df_original['Month'].iloc[-prediction_split]
    print(f'Beginning: {beginnning_value} at date: {beginning_date}')

    # determined predicted values
    total_dates = df_original.shape[0]
    total_prediction_range =  prediction_split
    count = 0
    previous_value = beginnning_value
    inverted = []
    for date in range(total_prediction_range):
        real_value = previous_value + predictions[date]
        inverted.append(real_value)
        previous_value = real_value

    # set all values before prediction start to zero
    zeros = [None for i in range(0,total_dates-prediction_split)]

    # append prediction results
    inverted_predictions = np.append(zeros,inverted)

    # tabulate
    df_results = pd.DataFrame(columns=['Date','Value','Pred Value'])
    df_results['Month'] = df_original['Month']
    df_results['Value'] = df_original['#Passengers']
    df_results['Pred Value'] = inverted_predictions
    df_results['Pred Value'][-prediction_split:].apply(lambda x: np.exp(x)) # inverting the log
    return df_results

def invert_first_difference(prediction_split,lag_window,predictions,df_original):
    # first real value to work from
    beginnning_value = df_original['#Passengers'].iloc[-prediction_split]
    beginning_date = df_original['Month'].iloc[-prediction_split]
    print(f'Beginning: {beginnning_value} at date: {beginning_date}')

    # determined predicted values
    total_dates = df_original.shape[0]
    total_prediction_range =  prediction_split
    count = 0
    previous_value = beginnning_value
    inverted = []
    for date in range(total_prediction_range):
        real_value = previous_value + predictions[date]
        inverted.append(real_value)
        previous_value = real_value

    # set all values before prediction start to zero
    zeros = [None for i in range(0,total_dates-prediction_split)]

    # append prediction results
    inverted_predictions = np.append(zeros,inverted)

    # tabulate
    df_results = pd.DataFrame(columns=['Date','Value','Pred Value'])
    df_results['Month'] = df_original['Month']
    df_results['Value'] = df_original['#Passengers']
    df_results['Pred Value'] = inverted_predictions

    return df_results