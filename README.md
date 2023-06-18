# Time series forecasting using ML techniques

This repository contains the work for my MEng mini-thesis. The project investigates feature engineering techniques for time series forecasting using machine learning. 

## Description repo

### Description of folders:
- misc code: old code, shared code or just useful snippets found during research. 
- old playground notebooks: messy jupyter notebooks used to trial new ideas or first attempt to develop various models and techniques. 
- pics: a place to save result figures.
- results: a place to save forecasting results.
- test data: example data used for the project

### Description of files:
- one_dimensional_time_series_forecasting.py: A python module containing a class which encapsulate the whole time series forecasting pipeline. Time series to supervised ml conversion, testing-training splits, training and tuning multiple ML and DL models, evaluating performance and plotting results. There are also a few miscellanious but usefull helper functions.
- basic_feature_engineering.ipynb: A notebook which investigates:
  - normalisation
  - differencing
  - log-differencing
- spectral_feature_engineering.ipynb: A notebook which investigates spectral methods used to denoise time series signals.

## Using the module
To use this work you will be primarily interested in the generalised forecasting pipeline implemented in the 
one_dimensional_time_series_forecasting.py module. 

All dependencies required are contained within:
- requirements.txt

And can be installed using pip:

`pip install -r requirements.txt`