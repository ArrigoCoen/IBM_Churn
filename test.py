


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


#%%

df = pd.read_csv("input/Telco-Customer-Churn.csv")
df.head()


#%%
df.info() # Also we could use 'df.isnull().sum()'





#%%

X = df.drop('Churn', axis=1)
y = df.Churn
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)




#%%


import os

os.getcwd()

#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer  # new in 0.20
from sklearn.compose import make_column_selector     # new in 0.22
ohe = OneHotEncoder()
ct = make_column_transformer((ohe, make_column_selector(dtype_include='object')), remainder = 'passthrough')


ct.fit_transform(X_train)

#%%

ct = make_column_transformer((ohe, make_column_selector(dtype_exclude='number')))

#%%

sns.distplot(df['charges'])

#%%

my_palette = ['colorblind', 'deep', 'pink', 'magma'][0]
sns.catplot(x="smoker", kind="count",hue = 'sex', palette=my_palette, data=df)

#%%

import os

os.getcwd()

#%%


#%%
column_trans.fit(X_train)



#%%
column_trans.get_feature_names_out()

#%%


model = grid_RF.best_params_['model']


#%%

#Feature ranking...
feature_list = column_trans.get_feature_names_out()
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
# print(feature_imp)
print(feature_imp.to_string())


#%%
satehu ohenc = OneHotEncoder();
ohenc.fit(xtrain_lbl);


ohenc.get_features_names(); x_cat_df.columns = ohenc.get_feature_names()

#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%

