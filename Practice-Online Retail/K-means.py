# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:47:34 2019

@author: HP
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')
df = pd.read_excel("Online_Retail.xlsx")
df.head()
df1 = df


# In[3]:


df1


# In[4]:


df1.Country.nunique()


# In[5]:


df1.Country.unique()


# In[6]:


customer_country=df1[['Country','CustomerID']].drop_duplicates()
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[7]:


df1 = df1.loc[df1['Country'] == 'United Kingdom']


# In[8]:


df1.isnull().sum(axis=0)


# In[9]:


df1 = df1[pd.notnull(df1['CustomerID'])]


# In[10]:


df1.isnull().sum(axis=0)


# In[11]:


df1 = df1[pd.notnull(df1['CustomerID'])]


# In[12]:


df1.Quantity.min()


# In[13]:


df1 = df1[(df1['Quantity']>0)]
df1.shape
df1.info()


# In[14]:


def unique_counts(df1):
   for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)
unique_counts(df1)


# In[15]:


df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']


# In[16]:


df1['InvoiceDate'].min()


# In[17]:


df1['InvoiceDate'].max()


# In[18]:


import datetime as dt
NOW = dt.datetime(2011,12,10)
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])


# In[19]:


rfmTable = df1.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalPrice': 'monetary_value'}, inplace=True)


# In[20]:


rfmTable.head()


# In[21]:


quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[22]:


segmented_rfm = rfmTable


# In[23]:


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[24]:


segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))


# In[25]:


segmented_rfm.head()


# In[26]:


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)


# In[27]:


segmented_rfm.head()


# In[28]:


segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)


# ## Clustering

# In[29]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[40]:


#normalize
normalized_rfm = rfmTable.iloc[:, 0:3]
normalized_rfm.recency = (normalized_rfm.recency - normalized_rfm.recency.mean())/normalized_rfm.recency.std()
normalized_rfm.frequency = (normalized_rfm.frequency - normalized_rfm.frequency.mean())/normalized_rfm.frequency.std()
normalized_rfm.monetary_value = (normalized_rfm.monetary_value - normalized_rfm.monetary_value.mean())/normalized_rfm.monetary_value.std()


# In[41]:


normalized_rfm.head(5)


# In[49]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(normalized_rfm)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow graph')
plt.xlabel('Cluster number')
plt.ylabel('WCSS')
plt.show()


# In[47]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(normalized_rfm)

rfmTable['clusters'] = clusters
rfmTable.head()


# In[48]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
#ax.scatter(normalized_rfm.recency, normalized_rfm.frequency, normalized_rfm.monetary_value, s=30)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
colors = ['blue', 'yellow', 'green', 'red']
for i in range(0,4):
    ax.scatter(normalized_rfm.recency[rfmTable.clusters == i], normalized_rfm.frequency[rfmTable.clusters == i], normalized_rfm.monetary_value[rfmTable.clusters == i], c = colors[i])
    


