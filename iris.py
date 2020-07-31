# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:39:23 2020

@author: harsh
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.preprocessing import MinMaxScaler, StandardScaler
mx= MinMaxScaler()
sc= StandardScaler()

X= mx.fit_transform(X)
X= sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y)

from sklearn.linear_model import LinearRegression, LogisticRegression
lin_r= LinearRegression()
log_r= LogisticRegression()

lin_r.fit(X_train, y_train)
lin_r.score(X_test, y_test)
lin_y= lin_r.predict(X)

log_r.fit(X_train, y_train)
log_r.score(X_test, y_test)
log_y= log_r.predict(X)

from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors= 5)

from sklearn.svm import SVC
sv= SVC()

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier()

kn.fit(X_train, y_train)
kn.score(X_test, y_test)
y_kn= kn.predict(X)

sv.fit(X_train, y_train)
sv.score(X_test, y_test)
y_sv= sv.predict(X)

nb.fit(X_train, y_train)
nb.score(X_test, y_test)
y_nb= nb.predict(X)

dtf.fit(X_train, y_train)
dtf.score(X_test, y_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm1= confusion_matrix(y, lin_y)
cm2= confusion_matrix(y, log_y)
cm3= confusion_matrix(y, y_kn)
cm4= confusion_matrix(y, y_sv)
cm5= confusion_matrix(y, y_nb)

precision_score(y, log_y, average='micro')
