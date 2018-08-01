# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ship_speed_fuel.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Predicting a new result with Linear Regression
y_pred_l= lin_reg.predict(X_test)

# Predicting a new result with Polynomial Regression
y_pred_p = lin_reg_2.predict(poly_reg.fit_transform(X_test))

lin_reg_2.predict(poly_reg.fit_transform([[1,15.4]]))

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred_p)



