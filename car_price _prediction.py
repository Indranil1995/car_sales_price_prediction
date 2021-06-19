#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())
print(df['fuel'].unique())


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_dataset=df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]


# In[10]:


final_dataset.head()


# In[11]:


final_dataset['Current Year']=2020


# In[12]:


final_dataset.head()


# In[13]:


final_dataset['no_year']=final_dataset['Current Year']-final_dataset['year']


# In[14]:


final_dataset.head()


# In[15]:


final_dataset.drop(['year'],axis=1,inplace=True)


# In[16]:


final_dataset.head()


# In[17]:


final_dataset.drop(['Current Year'],axis=1,inplace=True)


# In[18]:


final_dataset.head()


# In[19]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[20]:


final_dataset.head()


# In[21]:


import seaborn as sns


# In[22]:


sns.pairplot(final_dataset)


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[25]:


final_dataset.head()


# #Indipendent and Dependent features

# In[26]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[27]:


X.head()


# In[28]:


y.head()


# In[29]:


# Feature Importance 


# In[30]:


from sklearn.tree import ExtraTreeRegressor
model=ExtraTreeRegressor()
model.fit(X,y)
model.get_params()


# In[31]:


print(model.feature_importances_)


# In[32]:


# Plot graph for batter visualization


# In[33]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[35]:


X_train.shape


# In[36]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[37]:


# Hyperperameters


# In[38]:


import numpy as np
n_estimators =[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[39]:


from sklearn.model_selection import RandomizedSearchCV


# In[40]:


# Randomized Search CV

# Number of trees in random forest
n_estimators =[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#number of features to consider at every split
max_features= ["auto","sqrt"]
# Maximum number of levels in tree
max_depth= [int(x) for x in np.linspace(5,30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[41]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[42]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[43]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[44]:


rf_random.fit(X_train,y_train)


# In[45]:


predictions=rf_random.predict(X_test)


# In[46]:


predictions


# In[47]:


sns.distplot(y_test-predictions)


# In[48]:


plt.scatter(y_test,predictions)


# In[49]:


from sklearn import metrics


# In[50]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[51]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




