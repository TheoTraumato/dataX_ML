
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression





def plot_roc_curve(fitted_clf, x_test, y_test):
    """Plottet die ROC-Curve und gibt den AUC aus.

    :param fitted_clf (Classifier): Der Classifier, der bereits auf die Trainingsdaten gefitted wurde
    :param x_test (DataFrame): Die Testdaten der unabhöngigen Variablen
    :param y_test (Series): Die Testdaten der abhängigen Variable
    :return: None
    """
    probs = fitted_clf.predict_proba(x_test)
    probs = probs[:,1]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
