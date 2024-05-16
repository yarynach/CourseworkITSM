import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


from data import columns


ds = pd.read_csv("../data/new_data.csv")


# feature engineering
param_dict = pickle.load(open('../models/param_dict.pickle', 'rb'))


def impute_na(df, variable, value):
    return df[variable].fillna(value)


for column in columns.median_impute_columns:
    if column!='Life_expectancy':
        ds[column] = impute_na(ds, column, param_dict['median_impute_values'][column])


# impute with minus 1
for column in columns.impute_minus_one:
    ds[column] = impute_na(ds, column, param_dict['minus_one_impute_values'][column])



# Outlier Engineering
for column in columns.outlier_columns:
    ds[column] = ds[column].astype(float)
    out = np.where(ds[column] > param_dict['upper_lower_limits'][column + '_upper_limit'], True,
                   np.where(ds[column] < param_dict['upper_lower_limits'][column + '_lower_limit'], True, False))
    ds.loc[out, column] = np.nan

for column in columns.outlier_columns:
    ds[column] = impute_na(ds, column, param_dict['median_impute_outliers'][column])


for column in columns.cat_columns:
    ds[column] = ds[column].map(param_dict['map_dicts'][column])

    
# Масштабування
scaled_values = dict()
scaler = MinMaxScaler()
scaler.fit(ds[columns.columns_to_scale])
x_scaled = scaler.transform(ds[columns.columns_to_scale])
scaled_values = x_scaled
ds[columns.columns_to_scale] = pd.DataFrame(x_scaled, columns=columns.columns_to_scale)
ds.describe()

X = ds[columns.X_columns]

rf = pickle.load(open('../models/finalized_model.sav', 'rb'))

y_pred = rf.predict(X)

ds['Target_prediction'] = rf.predict(X)
ds.to_csv('prediction_results.csv', index=False)