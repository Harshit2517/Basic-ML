# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:11:51 2020

@author: harsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

col_names=['buying','maint','doors','persons','lug_boot','safety','Class']

from sklearn.datasets import load_car
dataset = load_car()
datset.columns= col_names

X= dataset.iloc[:,0:6]
y= dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()

y= lb.fit_transform(y)
lb.classes_
X= pd.get_dummies(X)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
mnx= MinMaxScaler()
sc= StandardScaler()
X= mnx.fit_transform(X)

X= sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test= train_test_split(X, y)

from sklearn.linear_model import LinearRegression, LogisticRegression
lin_r= LinearRegression()
log_r= LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

from sklearn.svm import SVC
sv= SVC()

from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier()


lin_r.fit(X_ttrain, y_train)
lin_r.score(X_temp, y)
lin_y= lin_r.predict(X_temp)

log_r.fit(X_temp, y)
log_r.score(X_temp, y)
log_y= log_r.predict(X_temp)

lin_r.fit(X_train, y_train)
log_r.fit(X_train, y_train)
knn.fit(X_train, y_train)
sv.fit(X_train, y_train)
nb.fit(X_train, y_train)
dtf.fit(X_train, y_train)

lin_r.score(X_test, y_test)
log_r.score(X_test, y_test)
sv.score(X_test, y_test)
knn.score(X_test, y_test)
nb.score(X_test, y_test)
dtf.score(X_test, y_test)


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm_log= confusion_matrix(y, log_y)

precision_score(y, log_y, average='micro')
recall_score(y, log_y, average='micro')
f1_score(y, log_y, average='micro')
