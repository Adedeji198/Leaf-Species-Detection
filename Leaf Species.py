#!/usr/bin/env python
# coding: utf-8

# LOADING LIBRARIES

# LEAF SPECIES DETECTION

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd #useful for loading the dataset
import numpy as np


# LOAD DATA SET

# In[2]:


from sklearn import datasets
datasets=load_iris()


# SUMMARISE DATA SET

# In[3]:


print(datasets.data)


# In[4]:


print(datasets.target)


# In[5]:


print(datasets.data.shape)


# SEGREGATE DATASET INTO X(INPUT/INDEPENT) AND Y(OUTPUT/DEPENDENT) VARIABLES

# In[6]:


X=pd.DataFrame(datasets.data,columns=datasets.feature_names)
X


# In[7]:


Y=datasets.target
Y


# SPLITTING DATA

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[9]:


print(X_train.shape)


# In[10]:


print(X_test.shape)


# FIND MAX-DEPTH VALUE

# In[11]:


accuracy=[]
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


for i in range (1, 10):
  model=DecisionTreeClassifier(max_depth=i, random_state=0)
  model.fit(X_train,y_train)
  pred=model.predict(X_test)
  score=accuracy_score(y_test,pred)
  accuracy.append(score)

  plt.figure(figsize=(12,6))
plt.plot(range(1,10), accuracy, color='red', linestyle='dashed', marker='o',
        markerfacecolor='green', markersize=10)
plt.title('Finding best Max_depth')
plt.xlabel('pred')
plt.ylabel('score')


# TRAINNING

# In[12]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)


# 

# MODEL PREDICTION

# In[13]:


y_pred =model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# ACCURACY SCORE

# In[14]:


from sklearn.metrics import accuracy_score
print("Accuracy of the model{0}%".format(accuracy_score(y_test,y_pred)*100))

