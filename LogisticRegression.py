#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math


# In[2]:


employeedataset_df = pd.read_csv('Datasets/employee_attrition_train.csv')


# In[3]:


employeedataset_df.head()


# In[4]:


#cant do for this dataset
#g=sns.pairplot(employeedataset_df)


# In[5]:


employeedataset_df.describe()


# In[6]:


sns.countplot(x="Attrition", data=employeedataset_df)


# In[7]:


sns.countplot(x="Attrition", hue="Department", data=employeedataset_df)


# In[8]:


employeedataset_df["Age"].hist()
plt.show()


# In[9]:


sns.countplot(x="Attrition", hue="OverTime", data=employeedataset_df)


# In[10]:


sns.countplot(x="Attrition", hue="JobLevel", data=employeedataset_df)


# In[4]:


df1 = employeedataset_df.dropna()


# In[5]:


#Machine Learning start
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model


# In[27]:


#x and y values, dummies changed Over Time to 0 and 1 instead of Yes and No
#X = df1[['Age', 'DistanceFromHome', 'OverTime', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StandardHours', 'TotalWorkingYears', 'WorkLifeBalance', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
X = df1[['Age', 'DistanceFromHome', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StandardHours', 'TotalWorkingYears', 'WorkLifeBalance', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
X = pd.get_dummies(X)
y = df1.Attrition


# In[28]:


X.columns


# In[29]:


X


# In[8]:


X.dtypes


# In[30]:


#split data
from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[32]:


# Fit Train model
model.fit(X_train, y_train)


# In[33]:


# accuracy score

model.score(X_test,y_test)


# In[34]:



model.score(X_train,y_train)


# In[35]:



X_test.head()


# In[36]:



y_test.head()


# In[37]:


#Predictions
predictions=model.predict(X_test)
predictions


# In[38]:


X_test[:5]


# In[39]:


y_test[:5]


# In[40]:


y_train.value_counts()


# In[21]:



from sklearn.metrics import classification_report


# In[22]:


pred = pd.DataFrame(predictions)


# In[23]:


pred.count()


# In[41]:


y_test[:10]


# In[42]:


print(classification_report(y_test,predictions))


# In[43]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix


# In[44]:



y_pred = model.predict(X_test)
confusion_matrix(y_test, predictions)


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


accuracy_score(y_test,predictions)


# In[34]:


import joblib

joblib.dump(model.fit(X_train, y_train),'logreg.mdl')


# In[ ]:




