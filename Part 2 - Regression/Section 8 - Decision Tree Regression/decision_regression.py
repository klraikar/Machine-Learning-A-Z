# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 20:48:35 2019

@author: Acer
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting the descision tree regressor to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor( random_state = 0 )
regressor.fit(X,y)

#predict the data
y_pred = regressor.predict(6.5)
print y_pred

#plot the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, regressor.predict(X_grid) , color = 'blue')
plt.show()


