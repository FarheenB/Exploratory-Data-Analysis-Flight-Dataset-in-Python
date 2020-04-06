#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[2]:


#get flights dataset
flights_data = pd.read_csv('dataset/FlightDelays.csv')


# In[3]:


#converting the object type(String type) to categorical data
#substitute to adding dummy variables

from collections import defaultdict
d = defaultdict(LabelEncoder)

#selecting cols that need to be transformed
df = pd.DataFrame(flights_data, columns = ['CARRIER', 'DEST', 'FL_DATE', 'ORIGIN','TAIL_NUM','Flight_Status'])

# Encoding the variable
fit = df.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
flights_df=pd.DataFrame(df.apply(lambda x: d[x.name].transform(x)))

#add the rest of the cols to the dataframe
flights_df['CRS_DEP_TIME']=flights_data['CRS_DEP_TIME']
flights_df['DEP_TIME']=flights_data['DEP_TIME']
flights_df['DISTANCE']=flights_data['DISTANCE']
flights_df['FL_NUM']=flights_data['FL_NUM']
flights_df['Weather']=flights_data['Weather']
flights_df['DAY_WEEK']=flights_data['DAY_WEEK']
flights_df['DAY_OF_MONTH']=flights_data['DAY_OF_MONTH']

#print top 5 values of the dataset
flights_df.head()


# # Exploratory Data Analysis

# In[4]:


#correlation matrix
corrmat = flights_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# In[5]:


#total no. of delayed and ontime flights

sns.countplot(x="Flight_Status", data=flights_data)
plt.show()


# In[6]:


#no. of delayed and ontime flights depending on carrier

sns.countplot(x="Flight_Status", hue="CARRIER",data=flights_data)
plt.show()


# In[7]:


#no. of delayed and ontime flights depending on destination

sns.countplot(x="Flight_Status", hue="DEST",data=flights_data)
plt.show()


# In[8]:


#no. of delayed and ontime flights depending on distance

sns.countplot(x="Flight_Status", hue="DISTANCE",data=flights_data)
plt.show()


# In[9]:


#no. of delayed and ontime flights depending on origin

sns.countplot(x="Flight_Status", hue="ORIGIN",data=flights_data)
plt.show()


# In[10]:


#no. of delayed and ontime flights depending on weather

sns.countplot(x="Flight_Status", hue="Weather",data=flights_data)
plt.show()


# In[11]:


#no. of delayed and ontime flights depending on day of the week

sns.countplot(x="Flight_Status", hue="DAY_WEEK",data=flights_data)
plt.show()


# In[12]:


#find no of delayed flights

Delayedflights = flights_data[(flights_data["Flight_Status"] == "delayed")]
Delayedflights.head()


# In[23]:


#histogram to find no of delayed flights due to departure time
sns.distplot(Delayedflights['DEP_TIME'])
plt.show()


# In[24]:


#histogram to find no of delayed flights due to day of the month
sns.distplot(Delayedflights['DAY_OF_MONTH'])
plt.show()


# In[20]:


#no. of times a tail number flight is delayed

sns.countplot(x="TAIL_NUM",data=Delayedflights)
plt.show()


# In[22]:


#no. of delayed flights on a particular date

sns.countplot(x="FL_DATE",data=Delayedflights)
plt.show()


# In[25]:


#scatterplot to get relation amongst all the variables
sns.set()
sns.pairplot(flights_df, size = 2.5)
plt.show()

