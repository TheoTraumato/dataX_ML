import data_prep
from sklearn import svm, metrics, model_selection

data_prep = data_prep.Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

params = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], C=[1.0, 0.5, 0.1, 0.05, 0.01],
              gamma=['scale', 'auto', 0.1, 1, 10, 20])



f1_scorer = metrics.make_scorer(metrics.f1_score)
grid_search = model_selection.GridSearchCV(estimator=svm.SVC, param_grid=params, verbose=2, scoring=f1_scorer,
                                           return_train_score=True, n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
best_kernel = [grid_search.best_params_['kernel']]
best_c = [grid_search.best_params_['C']]
best_gamma = [grid_search.best_params_['gamma']]

svm_clf = svm.SVC(kernel=best_kernel, C=best_c, gamma=best_gamma)
svm_clf.fit(x_train,y_train)

pred = svm_clf.predict(x_test)

# confusion matrix
print(metrics.confusion_matrix(y_test, pred))
# accuracy
print("acuracy:", metrics.accuracy_score(y_test, pred))
# precision score
print("precision:", metrics.precision_score(y_test, pred))
# recall score
print("recall", metrics.recall_score(y_test, pred))
#f1 score
print("F1-Score", metrics.f1_score(y_test, pred))
