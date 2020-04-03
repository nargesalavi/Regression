#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:30:15 2020

@author: Narges Alavi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Elimination import backElimination
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Catagorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid Dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#BackElimination
x_opt = backElimination(X_train, y_train, 0.05)

regressor = LinearRegression()
regressor.fit(x_opt, y_train)


plt.plot(y_train, color = 'red')
plt.plot(regressor.predict(x_opt), color = 'blue')
plt.title('50 Startups (Train set)')
plt.show()

# Visualising the Test set results
#plt.plot(y_test, color = 'red')
#plt.plot(y_pred, color = 'blue')
#plt.title('50 Startups (Test set)')
#plt.show()