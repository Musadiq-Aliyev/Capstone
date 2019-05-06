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


# In[4]:


df1


# In[5]:


df1.Country.nunique()


# In[6]:


df1.Country.unique()


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


def unique_counts(df1):
   for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)
unique_counts(df1)


# In[14]:


df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']


# In[15]:


df1['InvoiceDate'].min()


# In[16]:


df1['InvoiceDate'].max()


# In[17]:


import datetime as dt
NOW = dt.datetime(2011,12,10)
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])


# In[18]:


rfmTable = df1.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalPrice': 'monetary_value'}, inplace=True)


# In[19]:


quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[20]:


segmented_rfm = rfmTable


# In[21]:


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


# In[22]:


segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))


# In[23]:


segmented_rfm.head()


# In[24]:


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)


# In[25]:


segmented_rfm.head()


# In[26]:


segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)


# ## Clustering

# In[27]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[28]:


normalized_rfm = rfmTable.iloc[:, 0:3]
normalized_rfm.recency = (normalized_rfm.recency - normalized_rfm.recency.mean())/normalized_rfm.recency.std()
normalized_rfm.frequency = (normalized_rfm.frequency - normalized_rfm.frequency.mean())/normalized_rfm.frequency.std()
normalized_rfm.monetary_value = (normalized_rfm.monetary_value - normalized_rfm.monetary_value.mean())/normalized_rfm.monetary_value.std()


# In[29]:


normalized_rfm.head(5)


# In[30]:


import matplotlib.pyplot as plt
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(normalized_rfm, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[34]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(normalized_rfm)


# In[35]:


rfmTable['clusters'] = y_hc


# In[36]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

colors = ['blue', 'yellow', 'green', 'red', 'black']

for i in range(0,5):
    ax.scatter(normalized_rfm.recency[y_hc == i], normalized_rfm.frequency[y_hc == i], normalized_rfm.monetary_value[y_hc == i], c = colors[i])

