import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/dataset.csv")
#Заміна назв колонок, видалення пропусків
df.columns = [col.strip().replace(' ', '_') for col in df.columns]
X_columns = ['Country', 'Year', 'Status', 'Adult_Mortality',
       'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under-five_deaths', 'Polio', 'Total_expenditure',
       'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness__1-19_years',
       'thinness_5-9_years', 'Income_composition_of_resources', 'Schooling']

X_train, X_test = train_test_split(df,train_size=0.8, random_state=2004)

X_train.to_csv('data/train_data.csv', index=False)

X_test[X_columns].to_csv('data/new_data.csv', index=False)
