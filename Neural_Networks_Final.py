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

# In[2]:


y = data['Revenue'].copy()
X = data.drop('Revenue', axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.15, random_state=20)

print("Training dan Test Dataset")
# from sklearn.model_selection import train_test_split
# splitting the X, and y
# X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size = 0.2, random_state = 0)
# checking the shapes
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)


# In[3]:


#convert dataset from pandas frame to numpy dataset
y_train = y_train.values
#y_train


# # Neural Networks

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

X=X_train
Y=y_train

p = int(19) #features
n = len(y_train)
q = len(y_train) #number of classification problems

W = np.random.randn(p+1, q);


# In[5]:


## Train NN
Xb = np.hstack((np.ones((n,1)), X))
#q = np.shape(Y) #number of classification problems
M = 3 #number of hidden nodes

## initial weights
#W = np.random.randn(p+1, q);

alpha = 0.1 #step size
L = 10 #number of epochs

def logsig(_x):
    return 1/(1+np.exp(-_x))
        
for epoch in range(L):
    ind = np.random.permutation(n)
    for i in ind:
        # Forward-propagate 
        Yhat = logsig(Xb[[i],:]@W) 
         # Backpropagate
        #delta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
        delta = (Yhat-Y[[i]])*Yhat*(1-Yhat)
        Wnew = W - alpha*Xb[[i],:].T@delta
        W = Wnew
    print(epoch)


# In[6]:


Yhat=Yhat.T


# In[7]:


np.shape(Yhat)


# In[8]:


plt.scatter(X[:,0], X[:,1], c=Yhat[:,0])
plt.title('Predicted Labels, Neural Networks on training data')
plt.show()


# In[9]:


err_c1 = np.sum(abs(np.round(Yhat[len(Yhat)-1])-Y))
print('Errors, neural network classifier:', err_c1)


# In[10]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
#Calculating MSE, lower the value better it is. 0 means perfect prediction
#mse = mean_squared_error(np.round(Yhat[len(Yhat)-1]), Y)
mse = mean_squared_error(np.round(Yhat), Y)
print('Mean Squared Error, neural network classifier on training data:',np.round(mse*100,2))


# In[11]:


# validation test data
n2=len(y_test)
Xv = np.hstack((np.ones((n2,1)), X_test))


# In[12]:


An = np.hstack((np.ones((n2,1)), Xv@W))


# In[13]:


Hn = logsig(An)


# In[14]:


Yhatn = logsig(Xv@W)


# In[15]:


plt.scatter(X_test[:,0], X_test[:,1], c=Yhatn[:,0])
plt.title('Predicted Labels, Neural Networks on test data')
plt.show()


# In[16]:


err_c1 = np.sum(abs(np.round(Yhatn[len(Yhatn)-1])-y_train))
print('Errors, neural network classifier:', err_c1)


# In[17]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
#Calculating MSE, lower the value better it is. 0 means perfect prediction
#mse = mean_squared_error(np.round(Yhat[len(Yhat)-1]), Y)
mse = mean_squared_error(np.round(Yhat), Y)
print('Mean Squared Error, neural network classifier on training data:',np.round(mse*100,2))


# In[ ]:




