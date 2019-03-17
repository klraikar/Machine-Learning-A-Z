# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:30:58 2019

@author: Kiran Raikar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor_lin = LinearRegression()
regressor_lin.fit(X,y)

#fitting the Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #change the degree and check the predictions
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
regressor_lin2 = LinearRegression()
regressor_lin2.fit(X_poly,y)

#visualizing the Linear regression results
plt.scatter(X , y , color = 'red')
plt.plot(X , regressor_lin.predict(X) , color = 'blue') 
plt.plot(X , regressor_lin2.predict(X_poly) , color = 'green')
plt.title('Truth or Bluff (Polynomial Regression)')