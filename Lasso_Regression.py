import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

#TODO: One Hot Encoding wichtig oder nicht?

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# Lasso Regression:

lasso = Lasso(max_iter=1000)
# create dictionary of hyperparameters that we want optimize
parameters = {'alpha': np.logspace(-6, 6, 1000)}
# Searching for good alpha value on training data using 4-fold cross validation
regr = GridSearchCV(lasso, parameters, cv=4, scoring='neg_mean_squared_error')
# training on the train data
regr.fit(x_train, y_train)

# get the alpha value
best_alpha = regr.best_params_['alpha']
# retrain model on the whole training data with optimal alpha value
lasso = Lasso(alpha=best_alpha, max_iter=50000)
lasso.fit(x_train, y_train)
# evaluate model on test data
print(f"Mean Squared error for lambda={best_alpha}: {mean_squared_error(lasso.predict(x_test), y_test)}")