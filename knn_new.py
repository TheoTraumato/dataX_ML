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

# CV für Ermittlung des optimalen k
"""
Möglichst wenig giftige Pilze als nichtgiftig = wenig FN --> F1
"""
params = {"n_neighbors": range(1,50)}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params, scoring="f1", cv=5)
clf.fit(x_train, y_train)

best_k = clf.best_params_["n_neighbors"]
print(best_k)
print(f"{best_k}-Nearest Neighboirs achieved a score of {clf.best_score_}")


# KNN mit bestmöglichen K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("MSE=", mean_squared_error(y_pred, y_test))

#confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
K-Nearest Neighboirs achieved a score of 0.9367484780854021
MSE= 0.07040866568193008
"""