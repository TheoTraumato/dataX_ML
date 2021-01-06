import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#TODO: One Hot Encoding wichtig oder nicht?

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# Logistic:_Regression_Lasso

logistic_lasso = LogisticRegression(
    max_iter=5000000,
    penalty='l1',
    solver='saga',
    C= 0.1)

logistic_lasso.fit(x_train, y_train)
y_pred_lasso = logistic_lasso.predict(x_test)
print("Cross-Entropy = ", log_loss(y_pred_lasso, y_test))