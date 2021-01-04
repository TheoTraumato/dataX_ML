from data_prep import Data_Preperation
import sklearn
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_principalComponents(x_train, x_test, n_components):
    """Wendet Principal Component Analysis an und transformiert die Daten entsprechend. Die von den PCs nicht abgedeckte
    Varianz sollte möglichst klein sein.
    :param x_train: (DataFrame) Unabhängige Variablen für Training
    :param x_test: (DataFrame) Unabhängige Variablen für Test
    :return: principalComponents_train, principalComponents_test: Die principal Components für Training und Testdaten,
     gefittet nach den Trainingsdaten
    """
    pca = PCA(n_components)
    pca.fit(x_train)
    principalComponents_train = pca.transform(x_train)
    principalComponents_test = pca.transform(x_test)

    print('Anteil abgedeckte Varianz pro principal Component: ', pca.explained_variance_ratio_)
    print('Nicht abgedeckte Varianz: ', 1 - np.sum(pca.explained_variance_ratio_))
    return principalComponents_train, principalComponents_test


data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

pca_train, pca_test = get_principalComponents(x_train, x_test, 2)

pca_train_df = pd.DataFrame(data=pca_train, columns=['principal component ' + str(col_number + 1)
                                                     for col_number in range(len(pca_train[1]))])
pca_train_df.index = x_train.index
pca_train_df['y'] = y_train

print(pca_train_df.tail())

# Visualisierung wenn 2 PCAs behalten werden:
if len(pca_train[1]) == 2:
    plt.figure(figsize=(10, 10))
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.title("Principal Component Analysis of Mushroom Dateset")
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        plt.scatter(pca_train_df.loc[indicesToKeep, 'principal component 1'],
                    pca_train_df.loc[indicesToKeep, 'principal component 2'],
                    c=color, s=50)
    plt.legend(targets, prop={'size': 15})
    plt.show()

"""ERGEBNIS: Eher ungeeignet für unseren Datensatz, da zu viel Informationen verloren geht. Im ursprünglichen Datensatz
ist die Varianz sehr gleichmeißig auf alle Features verteilt, weswegen die Principal Components wenig Varianz
'an sich ziehen können'"""
