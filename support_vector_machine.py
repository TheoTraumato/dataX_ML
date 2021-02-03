from sklearn.model_selection import train_test_split

import data_prep
from sklearn import svm, metrics, model_selection
from Roc_curve import plot_roc_curve

from principal_component_analysis import get_principalComponents

data_prep = data_prep.Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run(oversampling=True)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123)

#x_train, x_test = get_principalComponents(x_train, x_test, 3)

# kernel: 'linear', 'poly', 'rbf', 'sigmoid'
params_linear = dict(kernel=[ 'linear'], C=[ 1.0, 0.1, 0.01, 0.001, 0.0001 ],)
params_poly = dict(kernel=['poly'], C=[ 0.1, 0.05, 0.01, 0.001 ],
                   gamma=['scale', 'auto',  0.5, 0.1, 0.01 ], degree=[2,3,4])
params_sigmoid = dict(kernel=['sigmoid'], C=[100, 10, 0.1 ],
                   gamma=['scale', 'auto',  1.0, 0.1, 0.01, 0.001, 0.0001 ])
params_rbf = dict(kernel=['rbf'], C=[10, 0.1, 0.05, 0.01, ],
                   gamma=['scale', 'auto', 10, 1.0,0.5, 0.1, 0.01 ])

# grid1: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
# grid2: {'C': 0.05, 'gamma': 'scale', 'kernel': 'linear'}
# grid3: (no more f1 scorer for gridsearch): {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
# grid4: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
# grid5: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
# grid6: {'C': 500, 'gamma': 0.001, 'kernel': 'rbf'}
# grid 7: {'C': 500, 'gamma': 0.001, 'kernel': 'rbf'} # accuracy 80,14 precision 70,42, recall 50, f1 58%

#linear kernel:
#grid 1: {'C': 0.01, 'kernel': 'linear'}
#grid 2: {'C': 0.01, 'kernel': 'linear'} # accuracy 79,97 precision 67,92%, recall 54,5% f1% 60%

#sigmoid_kernel: {'C': 0.1, 'gamma': 'scale', 'kernel': 'sigmoid'} acc: 79,52%, prec 66,83% recall 53,95%


#f1_scorer = metrics.make_scorer(metrics.f1_score)

'''grid_search = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=params_rbf, verbose=2,
                                           return_train_score=True, n_jobs=1)
grid_search.fit(x_train, y_train)
print('Mean cross-validated score of the best_estimator: ', grid_search.best_score_)
print('Best Estimator: ' , grid_search.best_estimator_)
print(grid_search.best_params_)'''


#svm_clf = svm.SVC(**grid_search.best_params_).fit(x_train, y_train)
svm_clf = svm.SVC(kernel='linear',  C=0.01, probability=True).fit(x_train, y_train)

val_pred = svm_clf.predict(x_test)


# confusion matrix
print(metrics.confusion_matrix(y_test, val_pred))
# accuracy
print("accuracy:", metrics.accuracy_score(y_test, val_pred))
# precision score
print("precision:", metrics.precision_score(y_test, val_pred))
# recall score
print("recall", metrics.recall_score(y_test, val_pred))
# f1 score
print("F1-Score", metrics.f1_score(y_test, val_pred))

plot_roc_curve(svm_clf, x_test, y_test)