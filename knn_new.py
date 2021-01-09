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
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params, scoring=LogLoss, cv=10)
clf.fit(x_train, y_train)

best_k = clf.best_params_["n_neighbors"]
print(f"{best_k}-Nearest Neighbors achieved a score of {clf.best_score_}")
score = clf.score(x_test, y_test)
print("Model score: ", score)

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
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("f1_score for y_pred", sklearn.metrics.f1_score(y_test, y_pred))
print("Cross-Entropy für y_pred= ", log_loss(y_pred, y_test))
print("Best_k = ", best_k)

"""
Output ohne PCA:
Model score:  -0.4608910175113216
[[1087  170]
 [ 234  265]]
f1_score for y_pred 0.5674518201284796
Cross-Entropy für y_pred=  7.946385403953634
Best_k =  95
"""