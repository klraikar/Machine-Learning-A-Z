# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:46:56 2019

@author: Kiran Raikar
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf' )
regressor.fit(X,y)

#predicting 
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
print sc_y.inverse_transform(np.array([[y_pred]]))

#visualzing the SVR Results
plt.scatter(X,y ,color = 'Red')
plt.plot(X , regressor.predict(X) , color = 'Blue')
plt.plot(y_pred,color = 'Black' )
plt.title('SVR Regression')
plt.xlabel('Salaries')
plt.ylabel('Level')
plt.show()
