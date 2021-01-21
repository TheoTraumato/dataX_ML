import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix, accuracy_score, \
    precision_score, recall_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from Confusion_Matrix import plot_confusion_matrix
from sklearn.metrics import f1_score

# Data preparation with One Hot Encoding:
import data_prep
from principal_component_analysis import get_principalComponents

data_prep = data_prep.Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(oversampling=True)

# Logistic Regression Basis Model:
logreg = LogisticRegression(penalty = "none")

# Train model:
logreg.fit(x_train, y_train)

# predict using x_test:
y_pred = logreg.predict(x_test)

# Evaluation Log Reg:#
print("Accuracy for basis model: ", accuracy_score(y_test, y_pred))
print("Precision for basis model: ", precision_score(y_test, y_pred))
print("Recall for basis model: ", recall_score(y_test, y_pred))
print("F1-score for basis model: ", sklearn.metrics.f1_score(y_test, y_pred))
logreg_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(logreg_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix basis model')

"""
Evaluation basis model:
Accuracy for basis model:  0.7997718197375927
Precision for basis model:  0.6766169154228856
Recall for basis model:  0.5517241379310345
F1-score for basis model:  0.6078212290502794
Confusion matrix, without normalization
[[1130  130]
 [ 221  272]]
"""

# Logistic Regression with Parameter Tuning:
logreg = LogisticRegression()

# create dictionary of hyperparameters that we want to optimize:
parameters_logreg = {"C" : np.logspace(-6, 6, 50), "penalty" : ['l1', "l2"], "solver" : ["liblinear"]}

# GridSearch
clf_logreg = GridSearchCV(logreg, parameters_logreg, scoring='neg_log_loss', cv=10)

# Train model:
clf_logreg.fit(x_train, y_train)

# predict using x_test:
y_pred_logreg = clf_logreg.predict(x_test)

# Evaluation Log Reg:#
print("best params for logreg: ", clf_logreg.best_params_)
print("Accuracy for logreg: ", accuracy_score(y_test, y_pred_logreg))
print("Precision for logreg: ", precision_score(y_test, y_pred_logreg))
print("Recall for logreg: ", recall_score(y_test, y_pred_logreg))
print("F1-score for logreg: ", sklearn.metrics.f1_score(y_test, y_pred_logreg))
logreg_matrix = confusion_matrix(y_test, y_pred_logreg)
plot_confusion_matrix(logreg_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix logreg optimized')

"""
Evaluation:
best params for logreg:  {'C': 0.244205309454865, 'penalty': 'l1', 'solver': 'liblinear'}
Accuracy for logreg:  0.8009127210496292
Precision for logreg:  0.6818181818181818
Recall for logreg:  0.5476673427991886
F1-score for logreg:  0.607424071991001
Confusion matrix, without normalization
[[1134  126]
 [ 223  270]]
"""

#

#

#

#lasso Regularisierung:

log_lass = LogisticRegression(max_iter=5000000, penalty='l1', solver='saga')

# create dictionary of hyperparameters that we want to optimize:
parameters_lasso = {"C" : np.logspace(-6, 6, 50)}

# Searching for good C value on training data using 10-fold cross validation
clf_lasso = GridSearchCV(log_lass, parameters_lasso, scoring='neg_log_loss', cv=10)

# training on the train data
clf_lasso.fit(x_train, y_train)

# get the alpha value and model score
print("best alpha for clf_lasso: ", clf_lasso.best_params_)

# predict using x_test:
y_pred_lasso = clf_lasso.predict(x_test)

#Evaluation:
print("Accuracy for clf_lasso: ", accuracy_score(y_test, y_pred_lasso))
print("f1_score for y_pred_lasso: ", sklearn.metrics.f1_score(y_test, y_pred_lasso))
lasso_matrix = confusion_matrix(y_test, y_pred_lasso)
plot_confusion_matrix(lasso_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix lasso')

"""
Evaluation lasso:
best alpha for clf_lasso:  {'C': 0.244205309454865}
Accuracy for clf_lasso:  0.8003422703936109
f1_score for y_pred_lasso:  0.6049661399548533
Confusion matrix, without normalization
[[1135  125]
 [ 225  268]]
"""

# Ridge Regularisierung:

log_ridge = LogisticRegression(max_iter=5000000, penalty='l2', solver='saga')

# create dictionary of hyperparameters that we want optimize
parameters_ridge = {"C" : np.logspace(-6, 6, 50)}

# Searching for good C value on training data using 10-fold cross validation
clf_ridge = GridSearchCV(log_ridge, parameters_ridge, scoring='neg_log_loss', cv=10)

# training on the train data
clf_ridge.fit(x_train, y_train)

# get the alpha value
print("best alpha for clf_ridge: ", clf_ridge.best_params_)

# predict using x_test:
y_pred_ridge = clf_ridge.predict(x_test)

# Evaluation:
print("Accuracy for clf_ridhe: ", accuracy_score(y_test, y_pred_ridge))
print("f1_score for y_pred_ridge: ", sklearn.metrics.f1_score(y_test, y_pred_ridge))
ridge_matrix = confusion_matrix(y_test, y_pred_ridge)
plot_confusion_matrix(ridge_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix ridge')

"""
Evaluation ridge:
best alpha for clf_ridge:  {'C': 0.07906043210907701}
Accuracy for clf_ridhe:  0.8003422703936109
f1_score for y_pred_ridge:  0.6040723981900452
Confusion matrix, without normalization
[[1136  124]
 [ 226  267]]
"""