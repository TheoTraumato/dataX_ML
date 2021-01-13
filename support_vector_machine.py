import data_prep
from sklearn import svm, metrics, model_selection

data_prep = data_prep.Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

#kernel: 'linear', 'poly', 'rbf', 'sigmoid'
params = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], C=[ 1.0,0.5, 0.1, 0.05, 0.01, ],
             gamma=['scale', 'auto',  1.0, 0.1, 0.01, ])
params_poly = dict(kernel=['poly', ], C=[ 1.0,0.5, 0.1, 0.05, 0.01, ],
             gamma=['scale', 'auto',  1.0, 0.1, 0.01, ])

#grid1: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}



f1_scorer = metrics.make_scorer(metrics.f1_score)
grid_search = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=params, verbose=2, scoring=f1_scorer,
                                           return_train_score=True, n_jobs=2)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
#best_kernel = [grid_search.best_params_['kernel']]
#best_c = [grid_search.best_params_['C']]
#best_gamma = [grid_search.best_params_['gamma']]

svm_clf = svm.SVC(**grid_search.best_params_)
svm_clf.fit(x_train, y_train)

pred = svm_clf.predict(x_test)

# confusion matrix
print(metrics.confusion_matrix(y_test, pred))
# accuracy
print("accuracy:", metrics.accuracy_score(y_test, pred))
# precision score
print("precision:", metrics.precision_score(y_test, pred))
# recall score
print("recall", metrics.recall_score(y_test, pred))
#f1 score
print("F1-Score", metrics.f1_score(y_test, pred))
