import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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
        """Liest Daten aus unseren Projektdaten, die von kaggle heruntergeladen wurden. Anschließend werden Duplikate
        entfernt, auf fehlende Werte (NaN) überprüft, und verschiedene Features mit String-Datentyp in numerische
        umgewandelt und die abhängige Variable von den unabhängigen getrennt.
         :return: x,y (DataFrame): Unabhängige Variablen, Abhängige Variablen
        """
        # Daten in DateFrame-Objekt lesen
        df = pd.read_csv('archive/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print(df)

        # CustomerID brauchen wir nicht, wird deswegen entfernt
        df = df.drop('customerID', axis=1)

        # Duplikate werden entfernt
        df.drop_duplicates(inplace=True)
        print(df.shape)

        # Alle Features die 'Yes' and 'No' als Ausprägungen haben werden umgewandelt
        boolean_values = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn' ]
        for column in boolean_values:
            df[column] = df[column].replace({'Yes': 1, 'No': 0})

        # Im Feature 'TotalCharges' sind die Werte als string gespeichert, werden hier umgewandelt
        df['TotalCharges'] = df['TotalCharges'].replace({" ":'0'})
        df['TotalCharges'] = df['TotalCharges'].astype(float)

        # Alle Features werden auf NaN-Werte überprüft (es sind keine vorhanden)
        print('Relative Menge an Missing Values: ', df.isna().sum() / (len(df)) * 100)

        # Abhängige wird von den unabhängigen Variablen gelöst
        y = df['Churn']
        x = df.drop('Churn', axis=1)

        return x, y

    def run(self, use_one_hot_encoding=True, standardize_data=True):
        """Liest die Daten und bereitet sie vor, wendet One Hot Encoding an falls gewünscht und trennt die Daten in
        Trainings- und Testdaten


        Args:
            use_one_hot_encoding: (Boolean) Soll One Hot Encoding angewandet werden auf dem DataFrame?
            standardize_data: (Boolean) Sollen die Daten standardisiert  werden?
        :return: x_train, x_test, y_train, y_test: Trainings- und Testdaten
        """
        x, y = self.__prepare_data()
        if use_one_hot_encoding:
            x = self.__one_hot_encoding(x)

        #Standardisieren
        #TODO: Prüfen ob Booleanwerte wirklich auch standardisiert werden
        if standardize_data:
            columns = x.columns
            index = x.index
            x_array = preprocessing.StandardScaler().fit_transform(x)
            x = pd.DataFrame(x_array, columns=columns)
            x.index = index

        print(x.head)
        #print(x.dtypes)


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)

        return x_train, x_test, y_train, y_test
