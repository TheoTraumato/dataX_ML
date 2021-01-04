from data_prep import Data_Preperation


def get_pca(x_train, x_test, y_train, y_test):
    """
    Wendet Principal Component Analysis an und transformiert die Daten entsprechend.
    :param x_train: (DataFrame) Unabhängige Variablen für Training
    :param x_test: (DataFrame) Unabhängige Variablen für Test
    :param y_train: (Series) Abhängige Variable für Training
    :param y_test: (Series) Abhängige Variable für Test
    :return: x_train, x_test, y_train, y_test: Gleiche Datentypen, aber transformiert nach PCA
    """
    pass


# Normaler Fall:
data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()
get_pca(x_train, x_test, y_train, y_test)

# Falls ihr kein One-Hot-Encoding verwenden wollt (z.B. für Decision Tree)
x_train, x_test, y_train, y_test = data_prep.run(use_one_hot_encoding=False)
