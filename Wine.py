# -*- coding: utf-8 -*-
"""
Created on Sat May 30 22:55:36 2020

@author: harsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


col_names=['Class','Alcohol','Malic_acid','Ash','Alcalinity','Magnesium','Total_phenols','Flavanoids','Nonflavanoid', 'Proanthocyanins','Color_intensity','Hue', 'dilutedwines','Proline']

from sklearn.datasets import load_wine
dataset = load_wine()
dataset.columns= col_names

X= dataset.iloc[:,1:15].values
y= dataset.iloc[:,0].values
temp= dataset.iloc[:,1:15].values
dataset.isnull().sum()

X= pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

sc= StandardScaler()
mnx= MinMaxScaler()
X_train= mnx.fit_transform(X_train)
X_train= sc.fit_transform(X_train)

from sklearn.linear_model import LinearRegression
lin_r= LinearRegression()
lin_r.fit(X_train, y_train)
lin_r.score(X_test, y_test)
lin_y= lin_r.predict(X_test)

from sklearn.linear_model import LogisticRegression
log_r= LogisticRegression()
log_r.fit(X_train, y_train)
log_r.score(X_test, y_test)
log_y= log_r.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors= 5)
kn.fit(X_train,y_train)
kn.score(X_test, y_test)
kn_y= kn.predict(X_test)

from sklearn.svm import SVC
sv= SVC()

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier()

lin_r.fit(X_train, y_train)
log_r.fit(X_train, y_train)
kn.fit(X_train, y_train)
sv.fit(X_train, y_train)
nb.fit(X_train, y_train)
dtf.fit(X_train, y_train)

lin_r.score(X_test, y_test)
log_r.score(X_test, y_test)
kn.score(X_test, y_test)
nb.score(X_test, y_test)
sv.score(X_test, y_test)
dtf.score(X_test, y_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm_lin= confusion_matrix(y, lin_y)

cm_log= confusion_matrix(y, log_y)

cm_kn= confusion_matrix(y, kn_y)

precision_score(y, log_y, average='micro')
precision_score(y, kn_y, average='micro')

recall_score(y, log_y, average='micro')
recall_score(y, kn_y, average='micro')

f1_score(y, log_y, average='micro')
f1_score(y, kn_y, average='micro')


