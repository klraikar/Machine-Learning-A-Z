# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:31:06 2019

@author: Acer
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#fitting the Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predecting the test set results
y_pred = regressor.predict(X_test)

#Visualizing the training set resuts
'''
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')'''

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test,y_pred, color = 'blue')
plt.title('Salary Vs Experience (Testing Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')