from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(standardize_data=True, oversampling=True)

# KNN
"""
Berechnet den Abstand eines neuen Datenpunktes x zu allen anderen Trainingsdatenpunkten.
K ist dabei die Anzahl der nächstgelegenen Punkte zu x.
Mittels Cross Validation und anhand des F1 Scores wird das optimale k bestimmt
"""

#basis modell:
# train model:
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_basis = knn.predict(x_test)

print("Accuracy for knn_basis: ", accuracy_score(y_test, y_pred_basis))
print("Precision for knn_basis: ", precision_score(y_test, y_pred_basis))
print("Recall for knn_basis: ", recall_score(y_test, y_pred_basis))

# CV für Ermittlung des optimalen k
params = {"n_neighbors": range(1,100)}
clf = GridSearchCV(knn, params, scoring="neg_log_loss", cv=10)
clf.fit(x_train, y_train)
best_k = clf.best_params_["n_neighbors"]

# train model:
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix
score = knn.score(x_test, y_test)
print("Accuracy for knn: ", accuracy_score(y_test, y_pred))
print("Precision for knn: ", precision_score(y_test, y_pred))
print("Recall for knn: ", recall_score(y_test, y_pred))
print("Best_k used = ", best_k)

# confusion matrix:
from Confusion_Matrix import plot_confusion_matrix
knn_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_matrix, classes=['churn=0','churn=1'],normalize= False,  title='Confusion matrix')

"""
Evaluation: standardize_data=False, oversampling=False:
Accuracy for knn:  0.7637010676156584
Precision for knn:  0.60828025477707
Recall for knn:  0.4775
Best_k used =  98
Confusion matrix, without normalization
[[882 123]
 [209 191]]
 
Evaluation: standardize_data=True, oversampling=False:
Accuracy for knn:  0.7622775800711744
Precision for knn:  0.6050955414012739
Recall for knn:  0.475
Best_k used =  76
Confusion matrix, without normalization
[[881 124]
 [210 190]] 
 
Evaluation: standardize_data=False, oversampling=True:
Accuracy for knn:  0.7615658362989324
Precision for knn:  0.5757575757575758
Recall for knn:  0.6175
Best_k used =  81
Confusion matrix, without normalization
[[823 182]
 [153 247]]
 
Evaluation: standardize_data=True, oversampling=True:
Accuracy for knn:  0.7601423487544484
Precision for knn:  0.5704697986577181
Recall for knn:  0.6375
Best_k used =  89
Confusion matrix, without normalization
[[813 192]
 [145 255]]
 """