#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier


# In[7]:


df = pd.read_csv('ChanDarren_RaiTaran_Lab2a.csv')


# In[8]:


df.sample(5)


# In[9]:


df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[10]:


df.head(5)


# In[12]:


#split train test 
x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)


# In[13]:


x_train.shape


# In[14]:


x_test.shape


# In[15]:


x_train


# In[18]:


#imputation transformare -- simple imputer is to used to missing value
trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')


# In[20]:


#one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')


# In[23]:


trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])


# In[27]:


#feature selection
trf4 = SelectKBest(score_func=chi2,k=8)


# In[28]:


trf5 = DecisionTreeClassifier()


# # create pipeline

# In[29]:


pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])


# # pipeline Vs make_pipeline
# ### pipeline requires naming of steps ,make_pipeline does not
# ### (Same applies to ColumnTransformer vs make_column_transformer

# In[30]:


#Alternate syntax
pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)


# In[33]:


#train
pipe.fit(x_train,y_train)


# # Explore the pipeline

# In[32]:


from sklearn import set_config
set_config(display='diagram')


# In[34]:


#predict
y_pred = pipe.predict(x_test)


# In[35]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# # Cross validation using pipeline

# In[37]:


#cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(pipe,x_train,y_train,cv=5,scoring='accuracy').mean()


# # Exploring the pipeline

# In[48]:



#export
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




