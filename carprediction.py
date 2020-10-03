import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\Virtual_Env\Car_Prediction\venv\car data.csv')
df.head(3)
df.shape
df.info()

print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


df['Car_Name'].unique()

## Check missing or null values
df.isnull().sum()

df.describe()
df.columns


## Remove car_name which is not a big factor
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head(2)

final_dataset['Current_Year'] = 2020
final_dataset.head(3)


## Calculate how many yeaars car is old
final_dataset['no_year'] = final_dataset['Current_Year'] - final_dataset['Year']
final_dataset.head(3)

final_dataset.drop(['Year'], axis=1, inplace=True)
final_dataset.head(3)

## Convert categorical variables in numeric by using dummy variable trap.
final_dataset = pd.get_dummies(final_dataset, drop_first =True)
final_dataset.head(3)


import seaborn as sns
sns.pairplot(final_dataset)


final_dataset=final_dataset.drop(['Current_Year'],axis=1)
final_dataset.head(3)
final_dataset.corr()

import matplotlib.pyplot as plt

## Find correlation between columns and plot it.
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))

sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')

## Select independent variables i.e all variables except price
X = final_dataset.iloc[:,1:]

## Select only dependent feature i.e price
y = final_dataset.iloc[:,0]

X['Owner'].unique()
X.head(3)
y.head(3)

## Feature importance i.e which feature is most necessary
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

## Plot graph of feature importances for better visualizations
feat_importances = pd.Series(model.feature_importances_, index = X.columns)

## Shows top 5 important features
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

## Split data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## Implement Random Forest Regressor & it uses decision trees and it does not require scaling values.
from sklearn.ensemble import RandomForestRegressor

rf_random = RandomForestRegressor()

## Hyperparameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop  = 1200, num = 12)]
print(n_estimators)

## Randomized Search CV

## Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop  = 1200, num = 12)]

## Number of features to consider at every split
max_features = ['auto', 'sqrt']

## Maximum number of levels in trees
max_depth = [int(x) for x in np.linspace(5,30, num=6 )]

## Max_depth.append(none)
## Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]

## Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]

## Hyperparameter tunning using randomized search CV
## RandomSearch CV:- It helps to do most parameters out of this considering how many estimators are there, how many max features is there & depth is there etc.

from sklearn.model_selection import RandomizedSearchCV

## Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

## Use the random grid to search for best hyperparameters
## First create the base model to tune
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='neg_mean_squared_error', n_iter = 10, cv =5, verbose =2, random_state=42, n_jobs = 1)

rf_random.fit(X_train, y_train)

rf_random.best_params_

rf_random.best_score_

prediction = rf_random.predict(X_test)

prediction

plt.scatter(y_test,prediction)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

import pickle

## open a file, where you are to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# Drop information to that file
pickle.dump(rf_random, file)

## Developed by Prashant Shukla
