import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

from data_analysis import correlation_matrix


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
        return pd.get_dummies(x, columns=dummy_list, drop_first=False)


    def __prepare_data(self):
        """Liest Daten aus unseren Projektdaten, die von kaggle heruntergeladen wurden. Anschließend werden Duplikate
        entfernt, auf fehlende Werte (NaN) überprüft, und verschiedene Features mit String-Datentyp in numerische
        umgewandelt und die abhängige Variable von den unabhängigen getrennt.
         :return: x,y (DataFrame): Unabhängige Variablen, Abhängige Variablen
        """
        # Daten in DateFrame-Objekt lesen
        df = pd.read_csv('archive/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        #print(df)

        # CustomerID, tenure und MonthlyCharges brauchen wir nicht, wird deswegen entfernt
        df = df.drop(['customerID'], axis=1)



        # Duplikate werden entfernt
        df.drop_duplicates(inplace=True)
        #print(df.shape)




        # Alle Features die 'Yes' and 'No' als Ausprägungen haben werden umgewandelt
        boolean_values = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn' ]
        for column in boolean_values:
            df[column] = df[column].replace({'Yes': 1, 'No': 0})



        # Im Feature 'TotalCharges' sind die Werte als string gespeichert, werden hier umgewandelt
        df['TotalCharges'] = df['TotalCharges'].replace({" ":'0'})
        df['TotalCharges'] = df['TotalCharges'].astype(float)

        df = self.__one_hot_encoding(df)

        #Correlations Matrix ausdrucken
        #correlation_matrix(df)
        print('Corr gender_Female: ', df['gender_Female'].corr(df['Churn']))
        print('Corr gender_Male: ', df['gender_Male'].corr(df['Churn']))

        #Gender haben beide Ausprägungen so gut wie keine Korrelation zu Churn -->Drop
        #Die ganzen 'No Internet Service' Ausprägungen sind bereits im Feature Internetservice enthalten-->Drop
        df = df.drop(['gender_Female', 'gender_Male', 'OnlineSecurity_No internet service',
                      'OnlineBackup_No internet service', 'DeviceProtection_No internet service',
                      'TechSupport_No internet service', 'StreamingTV_No internet service',
                      'StreamingMovies_No internet service', 'MultipleLines_No phone service'], axis=1)

        #correlation_matrix(df)

        #Durch MonthlyCharges und Tenure erhalten wir TotalCharges, überflüssig -->Drop
        #Die anderen Feats haben sehr geringe Korrelation mit Churn,
        # dafür teilweise hohe Korrelation mit MonthlyCharges -->Drop
        df = df.drop(['TotalCharges', 'StreamingMovies_Yes', 'StreamingTV_Yes', 'OnlineBackup_Yes', 'MultipleLines_Yes',
                      'MultipleLines_No', 'PhoneService', 'Partner', 'InternetService_DSL', 'InternetService_No',
                      'DeviceProtection_Yes'], axis=1)

        correlation_matrix(df)

        #TODO: Wieder verschmelzen zu binären Variablen


        #TODO: Binning von Tenure und Monthly Charges


        print(df.shape)

        # Alle Features werden auf NaN-Werte überprüft (es sind keine vorhanden)
        #print('Relative Menge an Missing Values: ', df.isna().sum() / (len(df)) * 100)

        # Abhängige wird von den unabhängigen Variablen gelöst
        y = df['Churn']
        x = df.drop('Churn', axis=1)

        return x, y

    def oversampling(self, x,y):
        """Oversampelt ein Dataset mithilfe des SMOTE-Algorithmus - Es werden nicht Einträge kopiert,
        sondern Einträge erstellt, die sehr ähnlich sind, um Overfitting entgegenzuwirken.

        :param x:(DataFrame) Unabhängige Variablen
        :param y: (Series) Abhängige Variablen
        :return: x,y: x,y nach Oversampling
        """
        # Hier wird gezeigt, dass unser Dataset imbalanced ist
        churn_val_count = y.value_counts(["Churn"])
        # print(churn_val_count)
        print('No: ', round(churn_val_count[0] / churn_val_count.sum() * 100, 2), ' %')
        print('Yes: ', round(churn_val_count[1] / churn_val_count.sum() * 100, 2), ' %')

        smote = SMOTE(sampling_strategy=0.5, k_neighbors=5)
        x,y = smote.fit_resample(x,y)

        churn_val_count = y.value_counts(["Churn"])
        print('Nach Oversampling')
        print('No: ', round(churn_val_count[0] / churn_val_count.sum() * 100, 2), ' %')
        print('Yes: ', round(churn_val_count[1] / churn_val_count.sum() * 100, 2), ' %')
        return x,y

    def run(self, use_one_hot_encoding=True, standardize_data=True, oversampling=True):
        """Liest die Daten und bereitet sie vor, wendet One Hot Encoding an falls gewünscht und trennt die Daten in
        Trainings- und Testdaten


        Args:
            use_one_hot_encoding: (Boolean) Soll One Hot Encoding angewandet werden auf dem DataFrame?
            standardize_data: (Boolean) Sollen die Daten standardisiert  werden?
        :return: x_train, x_test, y_train, y_test: Trainings- und Testdaten
        """
        x, y = self.__prepare_data()
        #if use_one_hot_encoding:
       #     x = self.__one_hot_encoding(x)

        #Standardisieren
        #TODO: Prüfen ob Booleanwerte wirklich auch standardisiert werden
        if standardize_data:
            columns = x.columns
            index = x.index
            x_array = preprocessing.StandardScaler().fit_transform(x)
            x = pd.DataFrame(x_array, columns=columns)
            x.index = index

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

        return x_train, x_test, y_train, y_test
