import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, make_scorer
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
clf = GridSearchCV(knn, params, scoring="f1", cv=10)
clf.fit(x_train, y_train)

# Evaluation Modell anhand dr Trainingsdaten:
best_k = clf.best_params_["n_neighbors"]
print(f"{best_k}-Nearest Neighbors achieved a score of {clf.best_score_}")

# KNN mit bestmöglichen K
"""
knn mit optimalem k
Gibt cross-validation heraus
Gibt confusion matrix heraus
Ziel: 
Möglichst geringes FN --> Möglichst wenig positive Kunden als negativ klassifizieren
Möglichst hohes TP, TN
--> Wollen möglichst viele Kunden die abwandern erkennen"""

#knn
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

#confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
score = knn.score(x_test, y_test)
print("Model Score for knn: ", score)
print("f1_score for y_pred", sklearn.metrics.f1_score(y_test, y_pred))
print("Cross-Entropy für y_pred= ", log_loss(y_pred, y_test))
print("Best_k used = ", best_k)
print(classification_report(y_test, y_pred))

# confusion matrix:
from Confusion_Matrix import plot_confusion_matrix
knn_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

"""
Trainings-Datensatz:
69-Nearest Neighbors achieved a score of 0.5737850753235719

Test-Datensatz:
Model Score for knn:  0.7763833428408443
f1_score for y_pred 0.5811965811965812
Cross-Entropy für y_pred=  7.723546524721654
Best_k used =  69
              precision    recall  f1-score   support

           0       0.83      0.86      0.85      1260
           1       0.61      0.55      0.58       493

    accuracy                           0.78      1753
   macro avg       0.72      0.71      0.71      1753
weighted avg       0.77      0.78      0.77      1753

Confusion matrix, without normalization
[[1089  171]
 [ 221  272]]
 """