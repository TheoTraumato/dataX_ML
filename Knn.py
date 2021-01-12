import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation
from principal_component_analysis import get_principalComponents

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# TODO: Features elimination, da KNN ungenau wird, je größer die Dimension

# PCA:

x_train, x_test = get_principalComponents(x_train, x_test, 2)

# KNN
"""
Berechnet den Abstand eines neuen Datenpunktes x zu allen anderen Trainingsdatenpunkten.
K ist dabei die Anzahl der nächstgelegenen Punkte zu x.
Zuerst wird für jedes K in range (1,40) der MSE visualisiert, um das beste K zu finden.
Schlussendlich wird das neue K in die KNN-Funktion eingesetzt
"""

error = []

# Berechnet den Fehler für alle K Werte zwischen 1 und 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# Visualisierung der Fehlerwerte
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

# KNN mit bestmöglichen K
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("MSE=", mean_squared_error(y_pred, y_test))