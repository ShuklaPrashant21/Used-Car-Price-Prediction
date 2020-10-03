import pandas as pd
import numpy as np


df = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\Virtual_Env\Car_Prediction\venv\car data.csv')


# In[3]:

df.head(3)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[7]:


df['Car_Name'].unique()


# In[8]:


## Check missing or null values
df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.columns


# In[11]:


## Remove car_name which is not a big factor
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[12]:


final_dataset.head(2)


# In[13]:

final_dataset['Current_Year'] = 2020
final_dataset.head(3)


# In[14]:


## Calculate how many yeaars car is old
final_dataset['no_year'] = final_dataset['Current_Year'] - final_dataset['Year']


# In[15]:


final_dataset.head(3)


# In[16]:


final_dataset.drop(['Year'], axis=1, inplace=True)


# In[17]:


final_dataset.head(3)


# In[18]:


## Convert categorical variables in numeric by using dummy variable trap.
final_dataset = pd.get_dummies(final_dataset, drop_first =True)


# In[19]:


final_dataset.head(3)


# In[20]:


import seaborn as sns


# In[21]:


sns.pairplot(final_dataset)


# In[22]:


final_dataset=final_dataset.drop(['Current_Year'],axis=1)


# In[24]:


final_dataset.head(3)


# In[25]:


final_dataset.corr()


# In[26]:


import matplotlib.pyplot as plt


# In[27]:


## Find correlation between columns and plot it.
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))

sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# In[28]:


## Select independent variables i.e all variables except price
X = final_dataset.iloc[:,1:]

## Select only dependent feature i.e price
y = final_dataset.iloc[:,0]


# In[31]:


X['Owner'].unique()


# In[29]:


X.head(3)


# In[30]:


y.head(3)


# In[32]:


## Feature importance i.e which feature is most necessary
from sklearn.ensemble import ExtraTreesRegressor


# In[33]:


model = ExtraTreesRegressor()
model.fit(X,y)


# In[34]:


print(model.feature_importances_)


# In[35]:


## Plot graph of feature importances for better visualizations
feat_importances = pd.Series(model.feature_importances_, index = X.columns)

## Shows top 5 important features
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[36]:


## Split data into test and train
from sklearn.model_selection import train_test_split


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)


# In[38]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[39]:


## Implement Random Forest Regressor & it uses decision trees and it does not require scaling values.
from sklearn.ensemble import RandomForestRegressor


# In[40]:


rf_random = RandomForestRegressor()


# In[41]:


## Hyperparameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop  = 1200, num = 12)]
print(n_estimators)


# In[42]:


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


# In[43]:


## Hyperparameter tunning using randomized search CV
## RandomSearch CV:- It helps to do most parameters out of this considering how many estimators are there, how many max features is there & depth is there etc.

from sklearn.model_selection import RandomizedSearchCV


# In[44]:


## Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[45]:


## Use the random grid to search for best hyperparameters
## First create the base model to tune
rf = RandomForestRegressor()


# In[46]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='neg_mean_squared_error', n_iter = 10, cv =5, verbose =2, random_state=42, n_jobs = 1)


# In[47]:


rf_random.fit(X_train, y_train)


# In[48]:


rf_random.best_params_


# In[49]:


rf_random.best_score_


# In[50]:


predictions = rf_random.predict(X_test)


# In[51]:


predictions


# In[52]:


sns.distplot(y_test - predictions)


# In[53]:


plt.scatter(y_test,predictions)


# In[54]:


from sklearn import metrics


# In[55]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[56]:


import pickle


# In[59]:


## open a file, where you are to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# Drop information to that file
pickle.dump(rf_random, file)

