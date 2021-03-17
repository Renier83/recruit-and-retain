#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy.random import seed
seed(1)
# Dependencies
import numpy as np
import pandas as pd
import tensorflow
tensorflow.keras.__version__


# In[2]:


df = pd.read_csv('Datasets/employee_attrition_train.csv')
employee_df = df.dropna()


# In[3]:


X = employee_df[['MonthlyIncome', 'TotalWorkingYears', 'Age', 'OverTime', 'NumCompaniesWorked', 'DistanceFromHome',
                 'YearsAtCompany']]
X = pd.get_dummies(X)
y = employee_df['Attrition']

print(X.shape, y.shape)


# In[4]:


X


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_scaler = MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# Step 1: Label-encode data set
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

# Step 2: Convert encoded labels to one-hot-encoding
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=8))
model.add(Dense(units=2, activation='softmax'))


# In[8]:


from tensorflow.keras.layers import Dense
number_inputs = 8
number_hidden_nodes = 4
model.add(Dense(units=number_hidden_nodes,
                activation='relu', input_dim=number_inputs))


# In[9]:


number_classes = 2
model.add(Dense(units=number_classes, activation='softmax'))


# In[10]:


model.summary()


# In[11]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=1000,
    shuffle=True,
    verbose=2
)


# In[13]:


y_pred = model.predict(X_test_scaled)
y_pred


# In[14]:


model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[20]:


import numpy as np
new_data = np.array([[20000, 20, 45, 1, 0,5, 25,5]])
print(f"Predicted class: {model.predict_classes(new_data)}")


# In[16]:


# Save the model in h5 format 
model.save("neural_model.h5")
print("Saved model to disk")


# In[17]:


#Save the model
# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save("neural_model.h5")
#print("Saved model to disk")

