# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 11:53:39 2020

@author: harsh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
import math

df= quandl.get('WIKI/GOOGL')

df['H-L']= (df['Adj. High']- df['Adj. Low'])/ df['Adj. Low'] *100
df['Change']= (df['Adj. Close']- df['Adj. Open'])/ df['Adj. Open'] *100

y= 'Adj. Close'
X= df[['H-L', 'Change', 'Adj. Volume', 'Adj. Close']]


df.isnull().sum()

forecast_out= int(math.ceil(0.005*len(X)))

X["Exp_Close"]= X[y].shift(-forecast_out)

from sklearn.impute import SimpleImputer

sim= SimpleImputer(missing_values= np.nan, strategy='mean')

X= sim.fit_transform(X)

X_temp= X[:,0:-1]
y= X[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X,y)

from sklearn.linear_model import LinearRegression
lin_r= LinearRegression()

lin_r.fit(X_train, y_train)
y_test=y_test.astype('int32')
y_predict =lin_r.predict(X_test).astype('int32')







