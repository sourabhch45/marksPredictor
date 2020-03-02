# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:35].values
y = dataset.iloc[:, 35:46].values




# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6,10,0,5,0,5,7,6,9,7,10,5,5,5,6,9,9,10,7,6,5,6,6,5,6,9,10,9,9,5,7,5,5,5]])
