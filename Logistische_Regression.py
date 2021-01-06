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

#lasso

lambdas = np.logspace(-6, 6, 50)
lasso = Lasso(max_iter=5000000)

parameters_lasso = {"alpha" : np.logspace(-6, 6, 50)}
clf_lasso = GridSearchCV(lasso, parameters_lasso, cv=10)
clf_lasso.fit(x_train, y_train)
print(clf_lasso.best_params_)
best_alpha_lasso = clf_lasso.best_params_["alpha"]

# Logistic:_Regression_Lasso

logistic_lasso = LogisticRegression(
    max_iter=5000000,
    penalty='l1',
    solver='saga',
    C= 1 / best_alpha_lasso)

logistic_lasso.fit(x_train, y_train)
y_pred_lasso = logistic_lasso.predict(x_test)
print("Cross-Entropy = ", log_loss(y_pred_lasso, y_test))

#Ridge

lambdas = np.logspace(-6, 6, 100)
ridge = (Ridge(max_iter=5000000))

parameters_ridge = {"alpha" : np.logspace(-6, 6, 50)}
clf_ridge = GridSearchCV(ridge, parameters_ridge, cv=10)
clf_ridge.fit(x_train, y_train)
print(clf_ridge.best_params_)
best_alpha_ridge = clf_ridge.best_params_["alpha"]

# Logistic:Ridge

logistic_ridge = LogisticRegression(
    max_iter=5000000,
    penalty='l1',
    solver='saga',
    C= 1 / best_alpha_ridge)

logistic_ridge.fit(x_train, y_train)
y_pred_ridge = logistic_ridge.predict(x_test)
print("Cross-Entropy = ", log_loss(y_pred_ridge, y_test))