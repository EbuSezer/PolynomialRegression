# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:17:46 2019

@author: Ebubekir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("age_height.csv",sep=";")
 
print(data.head())
y=data.Boy.values.reshape(-1,1)
x=data.Yas.values.reshape(-1,1)
plt.scatter(x,y)
plt.title("Age vs Height")
plt.xlabel("Age")
plt.ylabel("Height")

#%%Linear Regression
from sklearn.linear_model import LinearRegression
linear= LinearRegression()
linear.fit(x,y)

#making prediction about the x values
y_head=linear.predict(x)
plt.plot(x,y_head,color="red",label="LinearRegression")
plt.show()

#%%Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=2)

x_polynomial=polynomial_regression.fit_transform(x)

#%% fitting
linear_regression=LinearRegression()
linear_regression.fit(x_polynomial,y)   

#%%
y_head2=linear_regression.predict(x_polynomial)
plt.plot(x,y_head2,color="yellow",label="PolynomialRegression")
plt.legend()
plt.show()
