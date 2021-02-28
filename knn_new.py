from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from Roc_curve import plot_roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from Confusion_Matrix import plot_confusion_matrix

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(standardize_data=True, oversampling=True)

# KNN

#basis modell:
knn = KNeighborsClassifier()
knn_basis = knn.fit(x_train, y_train)
y_pred_basis = knn_basis.predict(x_test)

print("Accuracy for knn_basis: ", accuracy_score(y_test, y_pred_basis))
print("Precision for knn_basis: ", precision_score(y_test, y_pred_basis))
print("Recall for knn_basis: ", recall_score(y_test, y_pred_basis))

# confusion matrix basis:
knn_matrix = confusion_matrix(y_test, y_pred_basis)
plot_confusion_matrix(knn_matrix, classes=['churn=0','churn=1'],normalize= False,  title='Confusion matrix')

# CV für Ermittlung des optimalen k
params = {"n_neighbors": range(1,100)}
clf = GridSearchCV(knn, params, scoring="roc_auc", cv=10)
clf.fit(x_train, y_train)
best_k = clf.best_params_["n_neighbors"]

# train model:
knn = KNeighborsClassifier(n_neighbors=best_k)
knn_clf = knn.fit(x_train, y_train)
y_pred = knn_clf.predict(x_test)

# evaluation
score = knn.score(x_test, y_test)
print("Accuracy for knn: ", accuracy_score(y_test, y_pred))
print("Precision for knn: ", precision_score(y_test, y_pred))
print("Recall for knn: ", recall_score(y_test, y_pred))
print("Best_k used = ", best_k)
print("ROC for opt. model: ", plot_roc_curve(knn_clf, x_test, y_test))

# confusion matrix:
knn_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_matrix, classes=['churn=0','churn=1'],normalize= False,  title='Confusion matrix')
