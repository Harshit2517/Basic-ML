# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:30:59 2020

@author: harsh
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('student-por.csv', sep=';')

X= dataset.iloc[:,30:32].values
X_temp= dataset.iloc[:,0:33]
y= dataset.iloc[:,-1].values

dataset.isnull().sum()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
mx= MinMaxScaler()
sc= StandardScaler()

X_temp= pd.get_dummies(X_temp)

X_temp= mx.fit_transform(X_temp)

X_temp= sc.fit_transform(X_temp)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_temp, y)


from sklearn.linear_model import LinearRegression, LogisticRegression
lr= LinearRegression()
logr= LogisticRegression()

from sklearn.svm import SVC
sv= SVC()

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

lr.fit(X_train,y_train)
lr.score(X_test, y_test)
y_lr= lr.predict(X_test)

logr.fit(X_train,y_train)
logr.score(X_test, y_test)
y_log= logr.predict(X_test)

sv.fit(X_train,y_train)
sv.score(X_test, y_test)
y_sv= sv.predict(X_test)

nb.fit(X_train,y_train)
nb.score(X_test, y_test)
y_nb= nb.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm_lin= confusion_matrix(y_test, y_lr)
cm_log= confusion_matrix(y_test, y_log)
cm_sv= confusion_matrix(y_test, y_sv)
cm_nb= confusion_matrix(y_test, y_nb)