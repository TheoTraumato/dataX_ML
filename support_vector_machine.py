import data_prep
from sklearn import svm, metrics, model_selection

data_prep = data_prep.Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

params = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], C=[1.0, 0.5, 0.1, 0.05, 0.01], gamma=['scale', 'auto', 0.1, 1, 10, 20])

svm_clf = svm.SVC()
svm_clf.fit(x_train, y_train)
pred = svm_clf.predict(x_test)

# confusion matrix
print(metrics.confusion_matrix(y_test, pred))
# accuracy
print("acuracy:", metrics.accuracy_score(y_test, pred))
# precision score
print("precision:", metrics.precision_score(y_test, pred))
# recall score
print("recall", metrics.recall_score(y_test, pred))
