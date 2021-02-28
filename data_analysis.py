import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
import matplotlib.pyplot as plt


def correlation_matrix(df):
    """Berechne und plotte eine Korrelationsmatrix

    :param df (DataFrame): Das Dateframe anhand dessen die Matrix erstellt werden soll.
    Die Daten dürfen ausschließlich metrisch skaliert sein.
    :return: None
    """
    columns = df.columns
    index = df.index
    data = preprocessing.StandardScaler().fit_transform(df)
    df = pd.DataFrame(data, columns=columns)
    df.index = index

    print(df.corr())

    plt.figure(figsize=(25, 25))
    dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)



    # displaying heatmap
    mp.show()

def distribution(df, feat):
    """Stellt die Verteilung eines Features über den Datensatz, getrennt nach der unabhängigen Variable Churn, dar.

    :param df (DataFrame): Der Datensatz, die Werte des Features müssen metrisch skaliert sein.
    :param feat (String): Der Spaltenname des untersuchten Features
    :return: None
    """
    plt.figure()
    sb.distplot(df[df['Churn'] == 0][feat],  color='b', bins=20, label='No Churn')
    sb.distplot(df[df['Churn'] == 1][feat],  color='r', bins=20, label='Churn')
    plt.legend()
    plt.show()

def bar(df,feat):
    """Stellt die Häufigkeit eines Merkmals im Datensatz als Balkendiagramm dar.

    :param df (DataFrame): Der untersuchte Datensatz
    :param feat (String): Der Spaltenname des untersuchten Features
    :return: None
    """
    plt.figure()
    sb.countplot(data=df,x=feat)
    plt.legend()
    plt.show()

