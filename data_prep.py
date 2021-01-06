import pandas as pd
from sklearn.model_selection import train_test_split


class Data_Preperation():
    """
    Organisiert die Vorbereitung der Daten.

    Methods:
        run(use_one_hot_encoding=True)
        Liest die Daten, bereitet sie vor und gibt die bereinigten Trainings- und Testdaten zurück
    """

    def __one_hot_encoding(self, x):
        """
        Wendet one hot encoding auf den unabhängigen Variablen an. Da alle Features kategorisch sind,
         besteht die dummy list aus allen Features.
        :param x: (DataFrame) unabhängige Variablen
        :return: x (DataFrame) unabhängige Variablen, durch OHE in binäre Form umgewandelt.
        """
        # TODO:Prüfen ob boolean Features wie "Bruises" ausgelassen werden sollen - Jonas fragen!
        dummy_list = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection'
                      , 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        return pd.get_dummies(x, columns=dummy_list, drop_first=True)


    def __prepare_data(self):
        """
        Liest Daten aus unseren Projektdaten, die von kaggle heruntergeladen wurden. Anschließend werden Duplikate entfernt,
        auf fehlende Werte (NaN) überprüft und die abhängige Variable von den unabhängigen getrennt.
        :return: x,y (DataFrame): Unabhängige Variablen, Abhängige Variablen
        """
        # Read data
        df = pd.read_csv('archive/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print(df)

        # Data Preperation
        df = df.drop('customerID', axis=1)
        print(df.head)
        df.drop_duplicates(inplace=True)
        print(df.shape)

        print('Relative Menge an Missing Values: ', df.isna().sum() / (len(df)) * 100)

        y = df['class']
        x = df.drop('class', axis=1)

        y = y.replace({'p': 1, 'e': 0})

        return x, y

    def run(self, use_one_hot_encoding=True):
        """Liest die Daten und bereitet sie vor, wendet One Hot Encoding an falls gewünscht und trennt die Daten in
        Trainings- und Testdaten


        Args:
            use_one_hot_encoding: Soll One Hot Encoding angewandet werden auf dem DataFrame
        :return: x_train, x_test, y_train, y_test: Trainings- und Testdaten
        """
        x, y = self.__prepare_data()
        if use_one_hot_encoding:
            x = self.__one_hot_encoding(x)

        # TODO: Prüfen ob Outlier Häufigkeit der Merkmale nötig ist
        # TODO: Feature Selection mit PCA, Lasso-Regression, Ridge-Regression (Übung 4)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)

        return x_train, x_test, y_train, y_test
