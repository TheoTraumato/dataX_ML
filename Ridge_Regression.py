import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# Read data
df = pd.read_csv('archive/mushrooms.csv')
print(df)

# Data Preperation
df.drop_duplicates(inplace=True)

print('Relative Menge an Missing Values: ', df.isna().sum() / (len(df)) * 100)

y = df['class']
x = df.drop('class', axis=1)

y = y.replace({'p': 1, 'e': 0})

#One hot encoding
#dummy_list = x.columns.values
#x = pd.get_dummies(x, columns=dummy_list, drop_first=True)

# Ridge Regression:

lambdas = np.logspace(-6, 6, 100)
ridge = (Ridge(max_iter=5000000))

parameters = {"alpha" : np.logspace(-6, 6, 50)}
clf = GridSearchCV(ridge, parameters, cv=5)
clf.fit(x_train, y_train)
print(clf.best_params_)

best_alpha = clf.best_params_["alpha"]
ridge = Ridge(alpha=best_alpha, max_iter=5000000)
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
print("MSE=", mean_squared_error(y_pred, y_test))

errors = []

for l in lambdas:
  ridge.set_params(alpha=l)
  ridge.fit(x_train, y_train)
  errors.append(mean_squared_error(ridge.predict(x_test), y_test))

plt.figure(figsize=(10,6))
ax = plt.gca()
ax.plot(lambdas, errors)
ax.set_xscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("error")
plt.show()

"""
{'alpha': 1e-06}
MSE= 3.2238923765447616e-14
"""