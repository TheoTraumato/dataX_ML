import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(use_one_hot_encoding=True)

tree = DecisionTreeClassifier(criterion='entropy', random_state=10)
tree.fit(x_train, y_train)
labels = tree.predict(x_test)
print(f"F1-Score: {f1_score(labels, y_test)}")

tree.get_depth()

tree = DecisionTreeClassifier()
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 10), 'min_samples_split': range(2, 11, 2), 'min_samples_leaf': range(2, 17, 2)}
clf = GridSearchCV(tree, params, scoring='f1', cv=5)
clf.fit(x_train, y_train)
print(clf.best_params_, clf.best_score_)

c, md, mss, msl = clf.best_params_['criterion'], clf.best_params_['max_depth'], clf.best_params_['min_samples_split'], clf.best_params_['min_samples_leaf']
tree.set_params(criterion=c, max_depth=md, min_samples_split=mss, min_samples_leaf=msl)
tree.fit(x_train, y_train)
labels = tree.predict(x_test)
print(f"F1-Score: {f1_score(labels, y_test)}")


plt.figure(dpi=200)
plot_tree(tree, feature_names=x_train.columns)
plt.show()

forest = RandomForestClassifier(random_state=1)
params = {'n_estimators': range(10, 101, 10), 'criterion': ['entropy'], 'max_depth': [None], 'max_features': [0.6, 0.75, 0.8], 'max_samples': [0.5, 0.632, 0.75]}
clf = GridSearchCV(forest, params, scoring='f1', cv=5, verbose=2)
clf.fit(x_train, y_train)

n, c, md, mf, ms = clf.best_params_['n_estimators'], clf.best_params_['criterion'], clf.best_params_['max_depth'], clf.best_params_['max_features'], clf.best_params_['max_samples']
forest.set_params(n_estimators=n, criterion=c, max_depth=md, max_features=mf, max_samples=ms)
forest.fit(x_train, y_train)
labels = forest.predict(x_test)
print(f"F1-Score: {f1_score(labels, y_test)}")