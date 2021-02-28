import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score

from Roc_curve import plot_roc_curve
from data_prep import Data_Preperation


data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(oversampling=True)


""""Mini"baum - Kein Prepruning"""


tree = DecisionTreeClassifier(criterion='entropy', random_state=10)
tree.fit(x_train, y_train)
labels = tree.predict(x_test)


tree.get_depth()
plt.figure(dpi=400)

plot_tree(tree, feature_names=x_train.columns)  # Baum mit Kat.
plt.show()
y_pred = tree.predict(x_test)
print("Accuracy for basis model: ", accuracy_score(y_test, y_pred))
print("Precision for basis model: ", precision_score(y_test, y_pred))
print("Recall for basis model: ", recall_score(y_test, y_pred))
print("F1-score for basis model: ", sklearn.metrics.f1_score(y_test, y_pred))
plot_roc_curve(tree, x_test, y_test)



"""Optimierung der Parameter mit GridSearch"""

tree = DecisionTreeClassifier()
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 12), 'min_samples_split': range(2, 10, 2),
          'min_samples_leaf': range(2, 10, 2)}
clf = GridSearchCV(tree, params, scoring='recall', cv=5)
clf.fit(x_train, y_train)
print(clf.best_params_, clf.best_score_)

c, md, mss, msl = clf.best_params_['criterion'], clf.best_params_['max_depth'], clf.best_params_['min_samples_split'], \
                  clf.best_params_['min_samples_leaf']
tree.set_params(criterion=c, max_depth=md, min_samples_split=mss, min_samples_leaf=msl)
tree.fit(x_train, y_train)
labels = tree.predict(x_test)

plt.figure(dpi=400)
plot_tree(tree, feature_names=x_train.columns)  # Baum mit Kat.
plt.show()
y_pred = tree.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-score: ", sklearn.metrics.f1_score(y_test, y_pred))
plot_roc_curve(tree, x_test, y_test)


