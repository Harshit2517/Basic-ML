# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:22:51 2020

@author: harsh
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset= pd.read_csv('housing.csv')

dataset.isnull().sum()

X= dataset.iloc[:,[0,1,2,3,4,5,6,7,9]].values
y= dataset.iloc[:,8].values

from sklearn.impute import SimpleImputer
sim= SimpleImputer(strategy='mean')
X[:,[4]]= sim.fit_transform(X[:,[4]])
sim.statistics_

X= pd.DataFrame(X)
X.isnull().sum()

X= pd.get_dummies(X)


from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
X.iloc[:,-1]= lb.fit_transform(X.iloc[:,-1])
lb.classes_


from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc= StandardScaler()
mx= MinMaxScaler()
X= sc.fit_transform(X)
X=mx.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y)

from sklearn.linear_model import LinearRegression, LogisticRegression
lrs= LinearRegression()
log_r= LogisticRegression()

lrs.fit(X_train, y_train)
y_lin= lrs.predict(X_test)
lrs.score(X_test, y_test)

log_r.fit(X_train,y_train)
log_r.score(X_test, y_test)
log_y= log_r.predict(X_test)

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

nb.fit(X_train, y_train)
nb.score(X_test, y_test)
y_nb= nb.predict(X_test)

from sklear.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier()

dtf.fit(X_train, y_train)
dtf.score(X_test, y_test)
y_dtf= dtf.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cm_logistic= confusion_matrix(y_test, log_y)
cm_= confusion_matrix(y_test, y_nb)
cm_dtf= confusion_matrix(y_test, y_dtf)