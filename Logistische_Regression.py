import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from Confusion_Matrix import plot_confusion_matrix

#TODO: One Hot Encoding wichtig oder nicht?

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()


# logistic_regression:

logreg = LogisticRegression(C=0.01,solver = 'liblinear')
logreg.fit(x_train, y_train)
score = logreg.score(x_test, y_test)
print("Model Score for logreg: ", score)

# Evaluation logreg mit Cross-Entropy:
y_pred_logreg = logreg.predict(x_test)
print("Cross-Entropy for y_pred_logreg = ", log_loss(y_pred_logreg, y_test))
print("f1_score for y_pred_logreg", sklearn.metrics.f1_score(y_test, y_pred_logreg))
logreg_matrix = confusion_matrix(y_test, y_pred_logreg)
plot_confusion_matrix(logreg_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

"""
Ergebnis for logreg:
Model Score for logreg:  0.7801822323462415
Cross-Entropy for y_pred_logreg =  7.592342821547129
f1_score for y_pred_logreg 0.579520697167756
[[1104  153]
 [ 233  266]]
"""

#lasso_cv_paramter:
lambdas = np.logspace(-6, 6, 50)
lasso = Lasso(max_iter=5000000)

parameters_lasso = {"alpha" : np.logspace(-6, 6, 50)}
clf_lasso = GridSearchCV(lasso, parameters_lasso, cv=10) #scoring: log-loss?
clf_lasso.fit(x_train, y_train)
print("best alpha for lasso: ", clf_lasso.best_params_)
best_alpha_lasso = clf_lasso.best_params_["alpha"]

# Logistic:_Regression_Lasso:
logistic_lasso = LogisticRegression(
    max_iter=5000000,
    penalty='l1',
    solver='saga',
    C= 1 / best_alpha_lasso)

logistic_lasso.fit(x_train, y_train)
score = logistic_lasso.score(x_test, y_test)
print("Model Score for lasso: ", score)

# Evaluation Lasso mit Cross-Entropy:
y_pred_lasso = logistic_lasso.predict(x_test)
print("Cross-Entropy for y_pred_lasso = ", log_loss(y_pred_lasso, y_test))
print("f1_score for y_pred_lasso", sklearn.metrics.f1_score(y_test, y_pred_lasso))
print(confusion_matrix(y_test, y_pred_lasso))
lasso_matrix = confusion_matrix(y_test, y_pred_lasso)
plot_confusion_matrix(lasso_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

"""
Ergebnis für lasso:
best alpha for lasso:  {'alpha': 0.000868511373751352}
Model Score for lasso:  0.7881548974943052
Cross-Entropy for y_pred_lasso =  7.316975356671823
f1_score for y_pred_lasso 0.5912087912087912
[[1115  142]
 [ 230  269]]
"""

# Ridge_cv_parameter:
lambdas = np.logspace(-6, 6, 100)
ridge = (Ridge(max_iter=5000000))

parameters_ridge = {"alpha" : np.logspace(-6, 6, 50)}
clf_ridge = GridSearchCV(ridge, parameters_ridge, cv=10) #scoring: log-loss?
clf_ridge.fit(x_train, y_train)
print("Best alpha for ridge: ", clf_ridge.best_params_)
best_alpha_ridge = clf_ridge.best_params_["alpha"]

# Logistic_Regression_Ridge:
logistic_ridge = LogisticRegression(
    max_iter=5000000,
    penalty='l1',
    solver='saga',
    C= 1 / best_alpha_ridge)

logistic_ridge.fit(x_train, y_train)
score = logistic_ridge.score(x_test, y_test)
print("Model Score for ridge: ", score)

# Evaluation ridge mit Cross-Entropy:
y_pred_ridge = logistic_ridge.predict(x_test)
print("Cross-Entropy for y_pred_ridge = ", log_loss(y_pred_ridge, y_test))
print("f1_score for y_pred_ridge", sklearn.metrics.f1_score(y_test, y_pred_ridge))
print(confusion_matrix(y_test, y_pred_ridge))
ridge_matrix = confusion_matrix(y_test, y_pred_ridge)
plot_confusion_matrix(ridge_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

"""
Ergebnis für ridge:
Best alpha for ridge:  {'alpha': 68.66488450042998}
Model Score for ridge:  0.780751708428246
Cross-Entropy for y_pred_ridge =  7.572692028552994
f1_score for y_pred_ridge 0.5400238948626046
[[1145  112]
 [ 273  226]]
"""