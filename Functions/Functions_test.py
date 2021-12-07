#%%

from Functions/Project_Functions import *



#%%
import Project_Functions import *

file_name = "test"
x = 4
var = x
Project_Functions.my_piclke_dump(var, file_name)


#%%
import sys
sys.path.append('Functions')

from Project_Functions import *


#%%
import sys
sys.path.append('/Functions/Project_Functions.py')

# from Project_Functions import *
from Project_Functions import test_my_print
# import Project_Functions

file_name = "test"
x = 4
var = x
my_piclke_dump(var, file_name)

test_my_print()


#%%
import os
os.getcwd()

#%%


def test_my_print2():
    print("functionaaaaa!")
    return True

test_my_print2()

#%%




#%%



column_trans = make_column_transformer(
    (OneHotEncoder(), ['sex', 'smoker', 'region']),
    (StandardScaler(), ['age', 'bmi', 'children']),
    remainder='passthrough')

col_results = ['model', 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std']
models = [LinearRegression(), Lasso(), Ridge(), RandomForestRegressor(), XGBRegressor()]
# models = [LinearRegression(), Lasso(), Ridge()]
# models = [LinearRegression(), Lasso(), Ridge()]
results_df = pd.DataFrame(np.zeros([1,len(col_results)]), columns=col_results)

for i, model in enumerate(models):
    results_df = model_results(model, results_df)


results_df
#%%

# General computation modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import time
import datetime

import statsmodels
import os

# Data transformation modules
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Machine Learning modules
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

# Neural Networks modules
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor


# Possible useful modules
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# from sklearn.model_selection import cross_val_score

df = pd.read_csv("input/insurance.csv")
df.head()

X = df.drop('charges', axis=1)
y = df.charges
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



column_trans = make_column_transformer(
    (OneHotEncoder(), ['sex', 'smoker', 'region']),
    (StandardScaler(), ['age', 'bmi', 'children']),
    remainder='passthrough')

col_results = ['model', 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std']
models = [LinearRegression(), Lasso(), Ridge(), RandomForestRegressor(), XGBRegressor()]
# models = [LinearRegression(), Lasso(), Ridge()]
# models = [LinearRegression(), Lasso(), Ridge()]
results_df = pd.DataFrame(np.zeros([1,len(col_results)]), columns=col_results)

for i, model in enumerate(models):
    results_df = model_results(model, results_df)


results_df


#%%
print(zz)

#%%




#%%




#%%




#%%