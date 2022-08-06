import numpy as np
import pandas as pd 
import pickle

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

#importing the CSV here
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv')

#select target
target = df[['ICU Beds_x']] 

#data without medical resource
df_data = df.copy()
df_data.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1, inplace=True)

X = df_data
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Using a pipeline to dynamically standardize the data that goes into our model
 
pipeline1 = make_pipeline(StandardScaler(), Lasso(alpha=2))
pipeline1.fit(X_train, y_train) 

coef_list=pipeline1[1].coef_

#select only the coef dif of cero
loc=[i for i, e in enumerate(coef_list) if e != 0]

col_name=df_data.columns

best_columns = col_name[loc]

#regression model with the columns select
X_reg = X[best_columns]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y, test_size=0.3, random_state=45)

scaler = StandardScaler()
scaler.fit(X_train_reg)

X_train_reg = scaler.transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

filename = '../models/lin_reg.pkl'
pickle.dump(lin_reg, open(filename, 'wb'))
