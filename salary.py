# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:22:18 2020

@author: harsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'salary']


dataset= pd.read_csv('adult.csv', names=col_names, na_values= ' ?')


dataset.isnull().sum()

from sklearn.impute import SimpleImputer

sim= SimpleImputer(missing_values= np.nan, strategy='most_frequent')

# Numeric : [0, 2, 4, 10, 11, 12]
# Categorical : [1, 3, 5, 6, 7, 8, 9, 13, 14]

X= dataset.iloc[:,0:14]
y= dataset.iloc[:,14]



from sklearn.preprocessing import LabelEncoder

lb= LabelEncoder()
y= lb.fit_transform(y)


temp= X[['workclass', 'occupation', 'native-country']]

#temp['workclass'].value_counts()
#temp['occupation'].value_counts()
#temp['native-country'].value_counts()

temp= sim.fit_transform(temp)

temp= pd.DataFrame(temp)
temp.isnull().sum()

X[['workclass', 'occupation', 'native-country']]= temp

X.isnull().sum()


from sklearn.preprocessing import MinMaxScaler, StandardScaler
mx = MinMaxScaler()
sc = StandardScaler()
X= pd.get_dummies(X)


X= mx.fit_transform(X)
X= sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y)


from sklearn.linear_model import LinearRegression,LogisticRegression
lr= LinearRegression()

lr.fit(X_train, y_train)
lr.score(X_test, y_test)
lr_y= lr.predict(X_test)


logr= LogisticRegression()
logr.fit(X_train, y_train)
logr.score(X_test,y_test)
log_y= logr.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors=7)
kn.fit(X_train, y_train)
kn.score(X_test, y_test)
kn_y= kn.predict(X_test)

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

nb.fit(X_train, y_train)
nb.score(X_test, y_test)
nb_y= nb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_log= confusion_matrix(y_test, log_y)
cm_kn= confusion_matrix(y_test, kn_y)
cm_nb= confusion_matrix(y_test, nb_y)