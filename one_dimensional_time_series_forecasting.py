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

# predictive models
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# cross validation and hyper-parameter search
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

# class for one-dimensional time series forecasting
class time_series_prediction():

    def __init__(self,dates,one_d_time_series,lag_window_length,n_ahead_prediction):

        # raw input data + settings for time series -> supervised learning ML problem
        self.one_d_time_series = one_d_time_series#np.array(one_d_time_series)      # time series array, to array ensure index works as expected for class methods
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

        # cumprod results from predictions - set after calling .vis_results_time_series()
        self.real_vals_cumprod = None
        self.linear_reg_predictions_cumprod = None
        self.svm_predictions_cumprod = None
        self.neural_net_predictions_cumprod = None
    

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
        self.X_train = self.input_data[0:split,:]
        self.X_test = self.input_data[split:,:]
        self.y_train = self.target_data[0:split]
        self.y_test = self.target_data[split:]

        # generate different folds from training data for cross validation during hyperparameter tuning

        # different folds for cross validation
        tscv = TimeSeriesSplit(n_splits=5)

        # visualize cross validation splits
        fig,ax = plt.subplots(5,1,sharex=True)
        i = 0
        training_data = self.one_d_time_series[0:self.training_split]
        for tr_index, val_index in tscv.split(training_data): # training and validation splits for 5 folds
            # print(tr_index, val_index)
            ax[i].plot(tr_index,training_data[tr_index[0]:tr_index[-1]+1],'b-',label='training set')
            ax[i].plot(val_index,training_data[val_index[0]:val_index[-1]+1],'r-',label='validation set')
            ax[i].legend()
            i += 1
        ax[0].set_title('Cross validation sets for hyperparameter tuning')
        plt.tight_layout()
        plt.show()




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

        # save predictions
        self.linear_reg_predictions = predictions

    def support_vector_machine(self,model_tunning=True,C=None,kernel=None,epsilon=None):
        print('\nTraining support vector machine:')

        if model_tunning == False: #hyperparameter are known
            # train model
            svm_regres = SVR(max_iter=1000,C=C, kernel=kernel, epsilon=epsilon).fit(self.X_train,self.y_train)
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
            

            # save predictions
            self.svm_predictions = svm_predictions
        
        else: # must hyperparameter tune model

            # define model: support vector machine for regression
            model = SVR(max_iter=1000)

            # hyperparameter values to check
            param_grid = [
            {'C': [0.1, 1, 10, 100], 'kernel': ['linear','rbf','sigmoid'],'epsilon':[0.1,1,10,100]},
            ]

            # perform grid search, using cross validaiton
            tscv = TimeSeriesSplit(n_splits=5)
            gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_grid, scoring = 'neg_mean_squared_error',verbose=4,n_jobs=-1)
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

            # save predictions
            self.svm_predictions = svm_predictions

    def neural_net_mlp(self,verbose=0,model_tunning=True,hidden_layer_sizes=None,activation=None,learning_rate=None,learning_rate_init=None):
        print('\nTraining neural network: ')

        if model_tunning == False:
            # train neural network
            nn_regres = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,activation=activation,learning_rate=learning_rate,learning_rate_init=learning_rate_init,shuffle=False,random_state=1,max_iter=1000,verbose=verbose).fit(self.X_train,self.y_train)
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
        
        else: # perform hyperparameter tuning
            MLP = MLPRegressor(shuffle=False,max_iter=1000) # must set shuffle to false to avoid leakage of information due to sequance problem

            # hyperparameter values to check
            param_grid = [
            {'hidden_layer_sizes': [(10,),(100,),(1000,)], 'activation': ['logistic', 'tanh', 'relu'],'learning_rate': ['constant', 'invscaling', 'adaptive'], 'learning_rate_init':[0.001,0.01,1]}
 ]
            # perform grid search, using cross validaiton
            tscv = TimeSeriesSplit(n_splits=5)
            gsearch = GridSearchCV(estimator=MLP, cv=tscv, param_grid=param_grid, scoring = 'neg_mean_squared_error',verbose=4,n_jobs=-1)
            gsearch.fit(self.X_train, self.y_train)
            print('best_score: ', gsearch.best_score_)
            print('best_model: ', gsearch.best_estimator_)
            print('best_params: ',gsearch.best_params_)

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

    def naive_model(self): # t's prediction is t-1's value, note that this means you miss the first time point
        preds = np.zeros(len(self.one_d_time_series[self.training_split + self.lag_window_length:]))
        preds[0] = self.one_d_time_series[self.training_split + self.lag_window_length-1]
        preds[1:] = self.one_d_time_series[self.training_split + self.lag_window_length:-1]

        # evaluate
        mse = mean_squared_error(self.y_test,preds)
        mae = mean_absolute_error(self.y_test,preds)
        mape = mean_absolute_percentage_error(self.y_test,preds)

        print('\nNaive model results:')
        print('MAPE:',mape)
        print('RMSE: ',np.sqrt(mse))
        print('MAE: ',mae)

        self.naive_predictions = preds
        
# ****************************************************************************************************************
    # visualize results
# ****************************************************************************************************************
    def error(self,real_data,predicted_data):
        error = np.zeros(len(real_data))
        error = (real_data - predicted_data) / real_data
        return error

    # visualize orignal time series signal aswell as predictions    
    def vis_results_time_series(self,second_plot='error'):
        # plot prediction against actual + training data
        fig, ax = plt.subplots(2,1,figsize=(10,7),sharex=True)

        # original time series
        ax[0].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.one_d_time_series[self.training_split+self.lag_window_length:],'o-',linewidth=3,label='real values',markersize=5) 

        # predicted y values
        if self.linear_reg_predictions is not None:
            ax[0].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.linear_reg_predictions,'o-',label='linear regression prediction',markersize=5)
        ax[0].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.naive_predictions,'.--',label='naive prediction',markersize=5)
        if self.svm_predictions is not None:
            ax[0].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.svm_predictions,'.--',label='svm prediction',markersize=5)
        if self.neural_net_predictions is not None:
            ax[0].plot(self.time_series_dates[self.training_split+self.lag_window_length:],self.neural_net_predictions,'.--',label='nn prediction',markersize=5)

        ax[0].legend()
        ax[0].set_title('Real values vs model predictions')

        # plot error plot
        if second_plot == 'error':

            if self.linear_reg_predictions is not None:
                error_linreg = self.error(self.y_test,self.linear_reg_predictions)
            if self.svm_predictions is not None:
                error_svm = self.error(self.y_test,self.svm_predictions)
            if self.neural_net_predictions is not None:
                error_nn = self.error(self.y_test,self.neural_net_predictions)

            if self.linear_reg_predictions is not None:
                ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],error_linreg,'r-',label='linear reg error')
            if self.svm_predictions is not None:
                ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],error_svm,'-',label='svm error')
            if self.neural_net_predictions is not None:
                ax[1].plot(self.time_series_dates[self.training_split+self.lag_window_length:],error_nn,'-',label='nn error')
            
            ax[1].set_title('Error signal for predictive models')
            ax[1].set_xlabel('Dates')
            ax[1].legend()
            # ax[1].set_ylim([-10,10])
            ax[1].set_xticks([self.time_series_dates[x] for x in range(self.training_split,len(self.time_series_dates),28)])
            ax[1].tick_params(rotation=30)
        
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
    def test_train_plot(self):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(self.time_series_dates[0:self.training_split] ,self.one_d_time_series[0:self.training_split],'k-',label='Training data') # replace returns with sp_500 for other data plotting
        ax.plot(self.time_series_dates[self.training_split:] ,self.one_d_time_series[self.training_split:],'r-',label='Testing data')
        ax.plot(self.time_series_dates[self.training_split+self.lag_window_length:] ,self.y_test,'o',label='Windowed testing data') # important to match time by start 5 (length of time window) after where segmented our testing and training data
        plt.legend(loc=0) 
        ax.set_xticks([self.time_series_dates[x] for x in range(0,len(self.time_series_dates),150)])
        ax.tick_params(rotation=30) 
        ax.set_title('Test traing split')
        plt.tight_layout()

    # method to tabulate all results together nicely
    def results(self):
        df_results = pd.DataFrame(columns=['date','Value','Linear','SVM','NN','Naive'])
        df_results['date'] = self.time_series_dates
        df_results['Value'] = self.one_d_time_series
        
        # set all values before prediction start to zero
        zeros = [None for i in range(0,self.training_split+self.lag_window_length)]

        # append prediction results
        if self.linear_reg_predictions is not None:
            linear_predictions = np.append(zeros,self.linear_reg_predictions)
        if self.svm_predictions is not None:
            svm_predictions = np.append(zeros,self.svm_predictions)
        if self.neural_net_predictions is not None:
            nn_predictions = np.append(zeros,self.neural_net_predictions)
        naive_predictions = np.append(zeros,self.naive_predictions)

        # save predictions to df
        if self.linear_reg_predictions is not None:
            df_results['Linear'] = linear_predictions
        if self.svm_predictions is not None:
            df_results['SVM'] = svm_predictions
        if self.neural_net_predictions is not None:
            df_results['NN'] = nn_predictions
        df_results['Naive'] = naive_predictions
        
        return df_results

# this function determines whether predictions determine the correct movement for tomorrow.
def hit_rate(dates,original_values, predictions): # pass lists / arrays of dates, original values, and predictions
    # initialise dataframe
    df = pd.DataFrame(columns=['Date','Original Value','Daily PCT','Movement','Prediction','Predicted Movement'])

    # add known data as passed to function
    df['Date'] = dates.to_list()
    df['Original Value'] = original_values.to_list()
    df['Prediction'] = predictions.to_list()

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
    return df
