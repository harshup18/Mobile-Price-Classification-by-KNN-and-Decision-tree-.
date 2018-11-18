#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

ds=pd.read_csv('train.csv')
ds.head(5)


# In[2]:


ds.shape
ds.columns


# In[70]:


#ds.isnull().sum()
ds['price_range'].describe(), ds['price_range'].unique()


# In[6]:


ds.describe


# In[8]:


X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[9]:


ds = ds.drop(columns="blue")


# In[10]:


ds.columns


# In[11]:


X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[12]:


ds=ds.drop(columns='m_dep')


# In[13]:


X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[15]:


ds.columns


# In[20]:


ds['g'] = ds['three_g'] + ds['four_g']

ds=ds.drop(columns="three_g")
ds=ds.drop(columns="four_g")
ds.columns


# In[21]:


ds.columns


# In[22]:


X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[25]:


ds['siz'] = ds['sc_h'] * ds['sc_h'] + ds['sc_w'] * ds['sc_w']
ds['siz']=ds['siz'].apply(np.sqrt)
ds['siz']=ds['siz']/2.54
ds['siz']=ds['siz'].apply(np.around,decimals=1)
ds=ds.drop(columns="sc_h")
ds=ds.drop(columns="sc_w")
ds.columns


# In[26]:


X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[28]:


x=ds.iloc[:,:-1].values
x


# In[29]:


sns.jointplot(x='ram',y='price_range',data = ds )


# In[31]:


sns.distplot(ds['n_cores']) #1 - True, 0 - False


# In[33]:


sns.boxplot(x="price_range", y="talk_time", data=ds)


# In[37]:


myLabels = ['1 Cores' ,'2 Cores','3 Cores' , '4 Cores' , '5 Cores' , '6 Cores' , '7 Cores' , '8 Cores']
myValues = ds['n_cores'].value_counts().values
# print(myValues)
fig1, ax1 = plt.subplots()
ax1.pie(myValues , labels = myLabels, autopct='%1.1f%%', shadow=True ,startangle=90)
plt.show()


# In[44]:


labels4g = ["4G",'3G','2G']
values4g = ds['g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels=labels4g, autopct='%1.2f%%',shadow=True,startangle=90)
plt.show()


# In[46]:


X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[42]:


decisionTree = DecisionTreeClassifier(random_state = 40)
decisionTree.fit(X_train,Y_train)
decisionTree.score(X_test,Y_test)
                 


# In[47]:


prediction = KNN.predict(X_test)
matrix = confusion_matrix(Y_test,prediction)
print(matrix)


# In[5]:


plt.figure(figsize =(10,5))
sns.heatmap(matrix,annot=True)


# In[49]:


#Getting The Test Data
test_dataset = pd.read_csv('test.csv')

test_dataset['Network_Gen'] = test_dataset['four_g'] + test_dataset['three_g']

test_dataset['Dimension'] = test_dataset['sc_h']*test_dataset['sc_h'] + test_dataset['sc_w']*test_dataset['sc_w']
test_dataset['Dimension'] = test_dataset['Dimension'].apply(np.sqrt).apply(np.round)

del test_dataset['sc_h']
del test_dataset['sc_w']
del test_dataset['four_g']
del test_dataset['three_g']
del test_dataset['m_dep']
del test_dataset['blue']
del test_dataset['id']

test_dataset.shape


# In[50]:


#Getting The Test Data
test_dataset = pd.read_csv('test.csv')
predicted_price = KNN.predict(test_dataset)
predicted_price


# In[52]:


predicted_price


# In[53]:


#Adding predicted price to table
test_dataset['predicted_price'] = predicted_price
test_dataset.head(5)


# In[62]:


# ignore matplotlib warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
standardized_X = preprocessing.scale(X)
normalized_X = preprocessing.normalize(X)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[67]:





# In[69]:


warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
X = ds.drop('price_range',axis = 1)
Y = ds['price_range']
standard_X=StandardScaler()
X_train=standard_X.fit_transform(X_train)
X_test=standard_X.fit_transform(X_train)
#standardized_X = preprocessing.scale(X)
#normalized_X = preprocessing.normalize(X)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.30 , random_state = 42 )
X_train.shape
KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train , Y_train)
KNN.score(X_test,Y_test)


# In[7]:



#### Correlation Plot #### 

corrmat = ds.corr()
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(corrmat,vmax=0.8,square=True,annot=True,annot_kws={'size':8})


# In[72]:


display(ds.info())


# In[ ]:




