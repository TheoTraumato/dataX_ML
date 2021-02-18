import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from Confusion_Matrix import plot_confusion_matrix

# Data preparation with One Hot Encoding:
import data_prep

data_prep = data_prep.Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(standardize_data=True, oversampling=True)

# Logistic Regression Basis Model:
logreg = LogisticRegression(penalty="none")

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
plot_confusion_matrix(logreg_matrix, classes=['churn=0','churn=1'],normalize= False,  title='Confusion matrix basis model')

"""
Evaluation without oversampling:
Accuracy for basis model:  0.7814946619217081
Precision for basis model:  0.6514657980456026
Recall for basis model:  0.5
F1-score for basis model:  0.5657708628005658
Confusion matrix, without normalization
[[898 107]
 [200 200]]
 
Evaluation with oversampling:
Accuracy for basis model:  0.7665480427046263
Precision for basis model:  0.587378640776699
Recall for basis model:  0.605
F1-score for basis model:  0.5960591133004925
Confusion matrix, without normalization
[[835 170]
 [158 242]]
 
 
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
plot_confusion_matrix(logreg_matrix, classes=['churn=0','churn=1'],normalize= False,  title='Confusion matrix logreg optimized')

"""
Evaluation without oversampling:
best params for logreg:  {'C': 0.1389495494373136, 'penalty': 'l2', 'solver': 'liblinear'}
Accuracy for logreg:  0.7814946619217081
Precision for logreg:  0.6514657980456026
Recall for logreg:  0.5
F1-score for logreg:  0.5657708628005658
Confusion matrix, without normalization
[[898 107]
 [200 200]]

Evaluation with oversampling:
best params for logreg:  {'C': 0.1389495494373136, 'penalty': 'l2', 'solver': 'liblinear'}
Accuracy for logreg:  0.7672597864768683
Precision for logreg:  0.5883777239709443
Recall for logreg:  0.6075
F1-score for logreg:  0.5977859778597786
Confusion matrix, without normalization
[[835 170]
 [157 243]]
"""