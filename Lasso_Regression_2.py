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


#Trennen in Ziel- & unabhängige Variable:
df = pd.read_csv('archive/mushrooms.csv')
y = df['class']
x = df.drop('class', axis=1)

y = y.replace({'p': 1, 'e': 0})

#One hot encoding
dummy_list = x.columns.values
x = pd.get_dummies(x, columns=dummy_list, drop_first=True)

#Define columns
columns = x.columns

# Standardize
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


def get_coefficients(x, y, lambdas):
  lasso = Lasso(max_iter=5000000)
  lasso_coefs = []

  for alpha in lambdas:
    lasso.set_params(alpha=alpha)
    lasso.fit(x, y)
    lasso_coefs.append(lasso.coef_)

  return np.array(lasso_coefs)


def plot_lasso_coefs(lasso_coefs, lambdas, columns):
  plt.figure(figsize=(10, 6))
  ax = plt.gca()
  for i in range(lasso_coefs.shape[1]):
    ax.plot(lambdas, lasso_coefs[:, i], label=columns[i])
  ax.set_xscale('log')
  plt.xlabel('$\lambda$')
  plt.ylabel('weights')
  plt.legend()
  # plt.savefig('lasso_coef.png', dpi=300)
  plt.show()


if __name__ == '__main__':
  lambdas = np.logspace(-6, 6, 200)
  lasso_coefs = get_coefficients(x, y, lambdas)
  plot_lasso_coefs(lasso_coefs, lambdas, columns)