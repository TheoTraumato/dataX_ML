import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
import matplotlib.pyplot as plt


def correlation_matrix(df):
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