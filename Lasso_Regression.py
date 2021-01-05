import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#TODO: One Hot Encoding wichtig oder nicht?

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# Lasso Regression:

lambdas = np.logspace(-6, 6, 50)
lasso = Lasso(max_iter=5000000)

parameters = {"alpha" : np.logspace(-6, 6, 50)}
clf = GridSearchCV(lasso, parameters, cv=5)
clf.fit(x_train, y_train)
print(clf.best_params_)

"""{'alpha': 3.0888435964774785e-06}"""

# Modell mit best_alpha trainieren
best_alpha = clf.best_params_["alpha"]
lasso = Lasso(alpha=best_alpha, max_iter=5000000)
lasso.fit(x_train, y_train)
y_pred = lasso.predict(x_test)
print("MSE=", mean_squared_error(y_pred, y_test))

# Visualisierung
errors = []

for l in lambdas:
  lasso.set_params(alpha=l) #Regulierungsparameter
  lasso.fit(x_train, y_train)
  errors.append(mean_squared_error(lasso.predict(x_test), y_test))

plt.figure(figsize=(10,6))
ax = plt.gca()
ax.plot(lambdas, errors)
ax.set_xscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("error")
plt.show()

"""
{'alpha': 3.0888435964774785e-06}
MSE= 3.1690233535679233e-06
"""