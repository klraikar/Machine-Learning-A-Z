# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#handle the missing data 
#Ctrl+I for help for methods
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN' , strategy= 'mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding catgorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] =  labelencoder_X.fit_transform(X[:,0])  #encoding to 0 1 2

onehotencoder = OneHotEncoder(categorical_features= [0]) #one hot encoding is done since machine might think 2>1..
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y =  labelencoder_y.fit_transform(y) #one hot encoding is not needed for y as its has only 2 values
'''

#splitting the dataset 
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , y ,test_size = 0.2 , random_state = 42)
'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''