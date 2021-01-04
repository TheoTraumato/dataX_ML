import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

# Daten erstellen mit One Hot Encoding:
from data_prep import Data_Preperation

data_prep = Data_Preperation()
x_train, x_test, y_train, y_test = data_prep.run()

# Read data
df = pd.read_csv('archive/mushrooms.csv')
print(df)

# Data Preperation
df.drop_duplicates(inplace=True)

print('Relative Menge an Missing Values: ', df.isna().sum() / (len(df)) * 100)

y = df['class']
x = df.drop('class', axis=1)

y = y.replace({'p': 1, 'e': 0})

#One hot encoding
dummy_list = x.columns.values
x = pd.get_dummies(x, columns=dummy_list, drop_first=True)

# Ridge Regression:

model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(x, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)