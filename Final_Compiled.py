#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Data_full.csv')

#look for the missing values in each column
data.isna().sum()

# address missing data entries
data = data.dropna(axis=0).reset_index(drop=True)

# verify 
print("Total missing values:", data.isna().sum().sum())

{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

month_ordering = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
visitor_prefix = 'V'


data = ordinal_encode(data,'Month',month_ordering)
data = onehot_encode(data,'VisitorType',visitor_prefix)
data['Weekend'] = data['Weekend'].astype(np.int)
data['Revenue'] = data['Revenue'].astype(np.int)

data


# # Splitting into training data and evaluation data
# 

# In[2]:


y = data['Revenue'].copy()
X = data.drop('Revenue', axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=20)

print("Training dan Test Dataset")
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)


# In[3]:


#convert dataset from pandas frame to numpy dataset
y_train = y_train.values


# # Training & evaluating - First Classifier, Least Squares

# In[4]:


# Classifier 1 - Training Data
#w = (X^T X)^(-1)X^T y
X = X_train
y = y_train
w_train = np.linalg.inv(X.transpose()@X)@X.transpose()@y
#A = np.linalg.inv(X@X.T)

print(np.round(w_train,2))


# In[5]:


# all features
y_hat = np.sign(X_test@w_train)
print('considering all features', y_hat)


# In[6]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

print('Performance of Least-Squares based classifier')
print('')
mse = mean_squared_error(y_test, y_hat)
print('Mean squared error of testing set:', np.round(mse,4))
mae = mean_absolute_error(y_test, y_hat)
print('Mean absolute error of testing set:', np.round(mae,4))
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', np.round(rmse,4))


# # Training & evaluating - Second Classifier - Truncated SVD 

# In[7]:


min_err, min_r, min_w =np.inf,-1,None
err_sum = 0
for r in range(1,20):
    U, s, VT=np.linalg.svd(X_train)
    w = VT[:r, :].T@np.diag(1/s[:r])@U[:,:r].T@y_train
    err_ = np.mean(np.sign(X_test@w) != y_test)
    if err_<min_err:
        min_err, min_r, min_w = err_, r, w
                        
    err_sum+=np.mean(np.sign(X_train@min_w)!=y_train)


# In[8]:


# all features
y_hat = np.sign(X_test@min_w)
print('considering all features', y_hat)


# In[9]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

print('Performance of Truncated SVD based classifier')
print('')
mse = mean_squared_error(y_test, y_hat)
print('Mean squared error of testing set:', np.round(mse,4))
mae = mean_absolute_error(y_test, y_hat)
print('Mean absolute error of testing set:', np.round(mae,4))
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', np.round(rmse,4))


# # Training & evaluating - Third Classifier - Neural Networks (see independent code)

# # Optimization - Cross - Validation

# In[10]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Data_full.csv')

#look for the missing values in each column
data.isna().sum()

# address missing data entries
data = data.dropna(axis=0).reset_index(drop=True)

# verify 
print("Total missing values:", data.isna().sum().sum())

{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

month_ordering = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
visitor_prefix = 'V'


data = ordinal_encode(data,'Month',month_ordering)
data = onehot_encode(data,'VisitorType',visitor_prefix)
data['Weekend'] = data['Weekend'].astype(np.int)
data['Revenue'] = data['Revenue'].astype(np.int)

data


# In[11]:


import numpy as np
import scipy.io as sio
y = data['Revenue'].copy()
X = data.drop('Revenue', axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(X)


# In[12]:


#convert dataset from pandas frame to numpy dataset
h = y.values


# In[13]:


y = h


# In[14]:


y = data['Revenue'].copy()
X = data.drop('Revenue', axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=20)

print("Training dan Test Dataset")
# from sklearn.model_selection import train_test_split
# splitting the X, and y
# X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size = 0.2, random_state = 0)
# checking the shapes
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)


# In[15]:


err_sum = 0
for i in range(4):
    for j in range(4):
        if i == j: continue
        test_idx_1 = np.arange(i*3079, (i+1)*3079)
        test_idx_2 =np.arange(j*3079, (j+1)*3079)
        train_idx =np.setdiff1d(np.arange(12316), test_idx_1)
        train_idx =np.setdiff1d(train_idx, test_idx_2)
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test_1, y_test_1 =X[test_idx_1, :], y[test_idx_1]
        X_test_2, y_test_2 =X[test_idx_2, :], y[test_idx_2]
        min_err, min_r, min_w =np.inf,-1,None
        for r in range(1,20):
            U, s, VT=np.linalg.svd(X_train)
            w = VT[:r, :].T@np.diag(1/s[:r])@U[:,:r].T@y_train
            err_ = np.mean(np.sign(X_test_1@w) != y_test_1)
            if err_<min_err:
                min_err, min_r, min_w = err_, r, w
                        
            err_sum+=np.mean(np.sign(X_test_2@min_w)!=y_test_2)
                
print(err_sum/4/3)

