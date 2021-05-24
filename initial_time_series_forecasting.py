# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:09:23 2021

@author: tristan

First play around with time-series forecasting
"""

# import some shit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import some data
sp_500 = pd.read_csv('../test_data/GSPC.csv')
sp_500