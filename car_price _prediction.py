#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np


# In[42]:


df=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


# In[43]:


df.head()


# In[44]:


df.shape


# In[45]:


print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())
print(df['fuel'].unique())


# In[46]:


df.isnull().sum()


# In[47]:


df.describe()


# In[48]:


df.columns


# In[49]:


final_dataset=df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]


# In[50]:


final_dataset.head()


# In[51]:


final_dataset['Current Year']=2020


# In[52]:


final_dataset.head()


# In[53]:


final_dataset['no_year']=final_dataset['Current Year']-final_dataset['year']


# In[54]:


final_dataset.head()


# In[55]:


final_dataset.drop(['year'],axis=1,inplace=True)


# In[56]:


final_dataset.head()


# In[57]:


final_dataset.drop(['Current Year'],axis=1,inplace=True)


# In[58]:


final_dataset.head()


# In[59]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[60]:


final_dataset.head()


# In[61]:


import seaborn as sns


# In[62]:


sns.pairplot(final_dataset)


# In[63]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


final_dataset.head()


# #Indipendent and Dependent features

# In[65]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[66]:


X.head()


# In[67]:


y.head()


# In[68]:


# Feature Importance 


# In[69]:


from sklearn.tree import ExtraTreeRegressor
model=ExtraTreeRegressor()
model.fit(X,y)
model.get_params()


# In[70]:


print(model.feature_importances_)


# In[71]:


# Plot graph for batter visualization


# In[72]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[73]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[74]:


X_train.shape


# In[75]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y)


# In[76]:


y_pred = neigh.predict(X_test)


# In[77]:


from sklearn import metrics


# In[78]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[79]:


import pickle
# dump information to that file
pickle.dump(neigh, open('model.pkl','wb'))


# In[ ]:




