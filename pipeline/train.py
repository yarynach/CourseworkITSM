import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# custom files
from data import columns, model_best_hyperparameters

ds = pd.read_csv("../data/train_data.csv")


# feature engineering

def impute_na(df, variable, value):
    return df[variable].fillna(value)


# impute with median
median_impute_values = dict()
for column in columns.median_impute_columns:
    median_impute_values[column] = ds[column].median()
    ds[column] = impute_na(ds, column, median_impute_values[column])

# impute with minus 1
minus_one_impute_values = dict()
for column in columns.impute_minus_one:
    minus_one_impute_values[column] = -1
    ds[column] = impute_na(ds, column, minus_one_impute_values[column])
# викидування колонок
for column in columns.column_to_drop:
    ds.drop(column, axis=1, inplace=True)


# Outlier Engineering
def find_skewed_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary


upper_lower_limits = dict()
for column in columns.outlier_columns:
    upper_lower_limits[column + '_upper_limit'], upper_lower_limits[column + '_lower_limit'] = find_skewed_boundaries(
        ds, column, 2)
for column in columns.outlier_columns:
    out = np.where(ds[column] > upper_lower_limits[column + '_upper_limit'], True,
                   np.where(ds[column] < upper_lower_limits[column + '_lower_limit'], True, False))
    ds.loc[out, column] = np.nan

# Заміна шумів на медіану
median_impute_outliers = dict()
for column in columns.outlier_columns:
    median_impute_outliers[column] = ds[column].median()
    ds[column] = impute_na(ds, column, median_impute_outliers[column])

# Цілочисельне кодування
map_dicts = dict()
for column in columns.cat_columns:
    ds[column] = ds[column].astype('category')
    map_dicts[column] = dict(zip(ds[column], ds[column].cat.codes))
    ds[column] = ds[column].cat.codes

# Масштабування
scaled_values = dict()
scaler = MinMaxScaler()
scaler.fit(ds[columns.columns_to_scale])
x_scaled = scaler.transform(ds[columns.columns_to_scale])
scaled_values = x_scaled
ds[columns.columns_to_scale] = pd.DataFrame(x_scaled, columns=columns.columns_to_scale)
ds.describe()

# Збереження параметрів
param_dict = {'median_impute_values': median_impute_values,
              'minus_one_impute_values': minus_one_impute_values,
              'upper_lower_limits': upper_lower_limits,
              'map_dicts': map_dicts,
               'median_impute_outliers':median_impute_outliers
              }
with open('../models/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = ds[columns.X_columns]
y = ds[columns.y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
rf = RandomForestRegressor(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rankings = rf.feature_importances_.tolist()
importance = pd.DataFrame(sorted(zip(X_train.columns,rankings),reverse=True),columns=["variable","importance"]).sort_values("importance",ascending = False)

sns.barplot(x="importance",
            y="variable",
            data=importance[:-1])
plt.title('Variable Importance')
plt.tight_layout()
plt.savefig('variable_importance.png')



mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

dfsee=ds.head()
print('test set metrics:')
print('mae', mae)
print('mse', mse)
print("r2", r2)

filename = '../models/finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))
