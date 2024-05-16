mean_impute_columns = []

mode_impute_columns = []
median_impute_columns = ["Life_expectancy",
                         "Adult_Mortality",
                         "Polio",
                         "BMI",
                         "Diphtheria",
                         "thinness__1-19_years",
                         "thinness_5-9_years",
                         "Alcohol",
                         "Schooling",
                         "Income_composition_of_resources",
                         "Total_expenditure"
                         ]


random_impute_columns = []
categorical_unknown_columns= []

impute_zero=[]
impute_minus_one=["Hepatitis_B","GDP","Population"]
column_to_drop=["infant_deaths"]
outlier_columns = ['Schooling', 'Income_composition_of_resources']

cat_columns = ['Country','Status']

columns_to_scale = ['Adult_Mortality', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under-five_deaths', 'Polio', 'Total_expenditure',
       'Diphtheria', 'HIV/AIDS','Population', 'thinness__1-19_years',
       'thinness_5-9_years', 'Income_composition_of_resources', 'Schooling']

y_column = ['Life_expectancy'] # target variable
X_columns = ['Country', 'Year', 'Status', 'Adult_Mortality', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
       'Measles', 'BMI', 'under-five_deaths', 'Polio', 'Total_expenditure','GDP',
       'Diphtheria', 'HIV/AIDS','Population', 'thinness__1-19_years',
       'thinness_5-9_years', 'Income_composition_of_resources', 'Schooling']