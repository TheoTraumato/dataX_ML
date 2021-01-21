import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, make_scorer, accuracy_score, \
    precision_score, recall_score
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

#PCA:
#x_train, x_test = get_principalComponents(x_train, x_test, 2)

# KNN
"""
Berechnet den Abstand eines neuen Datenpunktes x zu allen anderen Trainingsdatenpunkten.
K ist dabei die Anzahl der nächstgelegenen Punkte zu x.
Mittels Cross Validation und anhand des F1 Scores wird das optimale k bestimmt
"""

# CV für Ermittlung des optimalen k
params = {"n_neighbors": range(1,100)}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params, scoring="neg_log_loss", cv=10)
clf.fit(x_train, y_train)
best_k = clf.best_params_["n_neighbors"]

# train model:
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix
score = knn.score(x_test, y_test)
print("Accuracy for knn: ", accuracy_score(y_test, y_pred))
print("Precision for knn: ", precision_score(y_test, y_pred))
print("Recall for knn: ", recall_score(y_test, y_pred))
print("Best_k used = ", best_k)
print(classification_report(y_test, y_pred))

# confusion matrix:
from Confusion_Matrix import plot_confusion_matrix
knn_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

"""
Accuracy for knn:  0.7792355961209355
Best_k used =  85
              precision    recall  f1-score   support

           0       0.83      0.87      0.85      1260
           1       0.62      0.54      0.58       493

    accuracy                           0.78      1753
   macro avg       0.73      0.71      0.72      1753
weighted avg       0.77      0.78      0.77      1753

Confusion matrix, without normalization
[[1098  162]
 [ 225  268]]
 """