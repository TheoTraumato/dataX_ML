import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from Confusion_Matrix import plot_confusion_matrix
from sklearn.metrics import f1_score

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# logistic_regression ohne regularisierung:
logreg = LogisticRegression(
    C=0.001,
    solver = 'liblinear')

# Train model:
logreg.fit(x_train, y_train)

#measuring model performance:
score = logreg.score(x_train, y_train)
print("Train-Score for logreg: ", score)

# predict using x_test:
y_pred_logreg = logreg.predict(x_test)

# Evaluation Log Reg:
print("Test-Score for logreg: ", accuracy_score(y_test, y_pred_logreg))
print("Cross-Entropy for y_pred_logreg: ", log_loss(y_pred_logreg, y_test))
print("F1-score for y_pred_logreg: ", sklearn.metrics.f1_score(y_test, y_pred_logreg))
logreg_matrix = confusion_matrix(y_test, y_pred_logreg)
plot_confusion_matrix(logreg_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

"""
Trainings-Datensatz:
Train-Score for logreg:  0.790144596651446

Test-Datensatz:
Test-Score for logreg:  0.7877923559612093
Cross-Entropy for y_pred_logreg:  7.329470821257711
F1-score for y_pred_logreg:  0.6331360946745561
Confusion matrix, without normalization
[[1060  200]
 [ 172  321]]
"""

#lasso Regularisierung:

log_lass = LogisticRegression(max_iter=5000000, penalty='l1', solver='saga')

# create dictionary of hyperparameters that we want to optimize:
parameters_lasso = {"C" : np.logspace(-6, 6, 50)}

# Searching for good C value on training data using 10-fold cross validation
clf_lasso = GridSearchCV(log_lass, parameters_lasso, cv=10)

# training on the train data
clf_lasso.fit(x_train, y_train)

# get the alpha value and model score
print("best alpha for clf_lasso: ", clf_lasso.best_params_)
print(f"Training-score for clf_lasso:  {clf_lasso.best_score_}")

# predict using x_test:
y_pred_lasso = clf_lasso.predict(x_test)

#Evaluation:
print("Test-Score for clf_lasso: ", accuracy_score(y_test, y_pred_lasso))
print("Cross-Entropy for y_pred_lasso: ", log_loss(y_pred_lasso, y_test))
print("f1_score for y_pred_lasso: ", sklearn.metrics.f1_score(y_test, y_pred_lasso))
lasso_matrix = confusion_matrix(y_test, y_pred_lasso)
plot_confusion_matrix(lasso_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix lasso')

"""
Ergebnis für lasso:
Trainings-Datensatz:
best alpha for clf_lasso:  {'C': 0.1389495494373136}
Training-score for clf_lasso:  0.8011823284446858

Test-Datensatz:
Test-Score for clf_lasso:  0.7986309184255562
Cross-Entropy for y_pred_lasso:  6.955145223057688
f1_score for y_pred_lasso:  0.6011299435028249
Confusion matrix, without normalization
[[1134  126]
 [ 227  266]]
"""

# Ridge Regularisierung:

log_ridge = LogisticRegression(max_iter=5000000, penalty='l2', solver='saga')

# create dictionary of hyperparameters that we want optimize
parameters_ridge = {"C" : np.logspace(-6, 6, 50)}

# Searching for good C value on training data using 10-fold cross validation
clf_ridge = GridSearchCV(log_ridge, parameters_ridge, cv=10)

# training on the train data
clf_ridge.fit(x_train, y_train)

# get the alpha value and model score
print("best alpha for clf_ridge: ", clf_ridge.best_params_)
print(f"Training-score for clf_ridge:  {clf_ridge.best_score_}")

# predict using x_test:
y_pred_ridge = clf_ridge.predict(x_test)

# Evaluation:
print("Test-Score for clf_ridhe: ", accuracy_score(y_test, y_pred_ridge))
print("Cross-Entropy for y_pred_ridge: ", log_loss(y_pred_ridge, y_test))
print("f1_score for y_pred_ridge: ", sklearn.metrics.f1_score(y_test, y_pred_ridge))
ridge_matrix = confusion_matrix(y_test, y_pred_ridge)
plot_confusion_matrix(ridge_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix ridge')

"""
Ergebnis für ridge:
Trainings-Datensatz:
best alpha for clf_ridge:  {'C': 0.04498432668969444}
Training-score for clf_ridge:  0.8013720803910918

Test-Datensatz:
Test-Score for clf_ridge:  0.7992013690815745
Cross-Entropy for y_pred_ridge:  6.935443011536021
f1_score for y_pred_ridge:  0.6009070294784581
Confusion matrix, without normalization
[[1136  124]
 [ 228  265]]
"""