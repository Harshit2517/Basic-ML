# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:26:00 2020

@author: harsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('bank-additional-full.csv', sep= ';')


dataset.isnull().sum()

X= dataset.iloc[:,0:20]

y= dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder

lb= LabelEncoder()

y= lb.fit_transform(y)
lb.classes_

X= pd.get_dummies(X)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
mx= MinMaxScaler()
sc= StandardScaler()

X_train= sc.fit_transform(X_train)

X_train= mx.fit_transform(X_train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y)


from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

nb.fit(X_train, y_train)
nb.score(X_test, y_test)
y_nb= nb.predict(X_test)

from sklearn.linear_model import LinearRegression, LogisticRegression
lin_r= LinearRegression()
log_r= LogisticRegression()

lin_r.fit(X, y)
lin_r.score(X, y)
lin_y= lin_r.predict(X)

log_r.fit(X_train, y_train)
log_r.score(X_test, y_test)
log_y= log_r.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cm_log= confusion_matrix(y_test, log_y)
cm_nb= confusion_matrix(y_test, y_nb)
