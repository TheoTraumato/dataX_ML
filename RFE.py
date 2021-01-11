import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#TODO: One Hot Encoding wichtig oder nicht?

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

#print(x_train.head)

def bla(x):
    test = []
    for i in x_train.to_string():
        if type(i) != int:
            test.append(i)
    return test

def colm(x):
    for col in x.columns:
        print(col)


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}:{df[column].unique()}')



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

#lasso_cv_paramter:
lambdas = np.logspace(-6, 6, 50)
lasso = Lasso(max_iter=5000000)

parameters_lasso = {"alpha" : np.logspace(-6, 6, 50)}
clf_lasso = GridSearchCV(lasso, parameters_lasso, cv=5) #scoring: log-loss?
clf_lasso.fit(x_train, y_train)
print("best alpha for lasso: ", clf_lasso.best_params_)
score = clf_lasso.score(x_test, y_test)
print("Model Score for lasso: ", score)
best_alpha_lasso = clf_lasso.best_params_["alpha"]

# Logistic:_Regression_Lasso:
lasso = Lasso(
    max_iter=5000000,
    alpha= best_alpha_lasso)

lasso.fit(x_train, y_train)
coeff_used_lasso = np.sum(lasso.coef_!=0)
print(coeff_used_lasso)

# Evaluation Lasso mit Cross-Entropy:
y_pred_lasso = lasso.predict(x_test)
print("Cross-Entropy for y_pred_lasso: ", log_loss(y_pred_lasso, y_test))
print("f1_score for y_pred_lasso: ", sklearn.metrics.f1_score(y_test, y_pred_lasso))
lasso_matrix = confusion_matrix(y_test, y_pred_lasso)
plot_confusion_matrix(lasso_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')



# Logistic:_Regression_Lasso: XXX
logistic_lasso = LogisticRegression(
    max_iter=5000000,
    penalty='l1',
    solver='saga',
    C= 1 / best_alpha_lasso)