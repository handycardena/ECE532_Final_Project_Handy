#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('Data_full.csv')


# In[3]:


#display data
data


# In[4]:


#gather data types of the different data entries found in the file
data.info()


# # Pre-processing

# In[5]:


#look for the missing values in each column
data.isna().sum()


# In[6]:


#display data corresponding to columns that are missing entries"
data[data.isna().sum(axis=1).astype(bool)]


# In[7]:


# address missing data entries
data = data.dropna(axis=0).reset_index(drop=True)


# # Changing string entries to numeric

# In[8]:


# verify 
print("Total missing values:", data.isna().sum().sum())

# print corrected data withou missing entries
# data


# In[9]:


{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}


# In[10]:


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


# In[11]:


month_ordering = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
visitor_prefix = 'V'
# encode data
data = ordinal_encode(data,'Month',month_ordering)
data = onehot_encode(data,'VisitorType',visitor_prefix)
data['Weekend'] = data['Weekend'].astype(np.int)
data['Revenue'] = data['Revenue'].astype(np.int)


# In[12]:


# display encoded data
data


# # Splitting into training data and evaluation data

# In[13]:


y = data['Revenue'].copy()
X = data.drop('Revenue', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=0.8, random_state=20)

print("Training data size Test Dataset")
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_eval :", X_eval.shape)
print("Shape of y_eval :", y_eval.shape)


# In[14]:


import numpy as np
# from spicy.io import loadmat
import matplotlib.pyplot as plt

#A = np.genfromtxt('Data_Raw.csv', delimiter=',')
#print(A.dtype)

#Data = np.genfromtxt('Data_null.csv', delimiter=',')
#x_all = Data[0:12330,0:14] # features
#y_train = Data[0:12330,14] # corresponding labels

#x_train = Data[0:12330,0:14] # features
#y_train = Data[0:12330,14] # corresponding labels

# evaluation data
#x_eval= Data[1001:12330,0:14] # features
#y_eval = Data[1001:12330,14] # corresponding labels

# X = Data[0:3,0:14]
# y = Data[:,14] 

# Classifier 1
#w = (X^T X)^(-1)X^T y
#X = x_train
#y = y_train
#w = np.linalg.inv(X.transpose()@X)@X.transpose()@y
#A = np.linalg.inv(X@X.T)

#print(np.round(w,2))


# # Training & evaluating - Least Squares

# In[15]:


# Classifier 1 - Training Data
#w = (X^T X)^(-1)X^T y
X = X_train
y = y_train
w_train = np.linalg.inv(X.transpose()@X)@X.transpose()@y
#A = np.linalg.inv(X@X.T)

print(np.round(w_train,2))


# In[16]:


# all features
print('considering all features')
y_hat = np.sign(X_eval@w_train)

#error_vec = [0 if i[0]==i[1] else 1 for i in np.hstack((y_hat, y_test))]
#print('Errors: '+ str(sum(error_vec)))
#print('Percent error: '+str(100.0*sum(error_vec)/len(error_vec))+'%')


# In[17]:


#print(np.round(y_hat,2))
#np.shape(y_hat)
#np.shape(y_eval)


# In[18]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

print('Performance of Least-Squares based classifier')
print('')
mse = mean_squared_error(y_eval, y_hat)
print('Mean squared error of testing set:', np.round(mse,4))
mae = mean_absolute_error(y_eval, y_hat)
print('Mean absolute error of testing set:', np.round(mae,4))
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', np.round(rmse,4))


# # Training & evaluating - Truncated SVD 

# In[19]:


import numpy as np
import scipy.io as sio

U, s, VT = np.linalg.svd(X_train,full_matrices=False)
#w = VT.T@np.diag(1/s)@U.T@y_train
#err_ = np.mean(np.sign(X_test@w) != y_test)


# In[20]:


#U, s, VT = np.linalg.svd(X_train)
np.shape(X_train)


# In[21]:


U.shape, s.shape, VT.shape


# In[22]:


#w = VT.T@np.diag(1/s)@U.T@y_train
w_svd = VT.T@np.diag(1/s)@U.T@y_train
print(np.round(w_svd,2))


# In[23]:


# all features
print('considering all features')
y_hat = np.sign(X_eval@w_svd)

#error_vec = [0 if i[0]==i[1] else 1 for i in np.hstack((y_hat, y_test))]
#print('Errors: '+ str(sum(error_vec)))
#print('Percent error: '+str(100.0*sum(error_vec)/len(error_vec))+'%')


# In[24]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
#Calculating MSE, lower the value better it is. 0 means perfect prediction

print('Performance of Truncated SVD based classifier')
print('')

mse = mean_squared_error(y_eval, y_hat)
print('Mean squared error of testing set:', np.round(mse,4))
mae = mean_absolute_error(y_eval, y_hat)
print('Mean absolute error of testing set:', np.round(mae,4))
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', np.round(rmse,4))


# In[25]:


data.info()

