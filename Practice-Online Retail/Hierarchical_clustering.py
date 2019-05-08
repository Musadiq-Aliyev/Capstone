#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')
d_set = pd.read_excel("Online_Retail.xlsx")
d_set.head()
d_set1 = d_set


# In[4]:


d_set1



# In[5]:


d_set1.Country.nunique()


# In[6]:


d_set1.Country.unique()


# In[7]:


d_set1 = d_set1.loc[d_set1['Country'] == 'United Kingdom']


# In[8]:


d_set1.isnull().sum(axis=0)


# In[9]:


d_set1 = d_set1[pd.notnull(d_set1['CustomerID'])]


# In[10]:


d_set1.isnull().sum(axis=0)


# In[11]:


d_set1 = d_set1[pd.notnull(d_set1['CustomerID'])]


# In[12]:


d_set1.Quantity.min()


# In[13]:


def unique_counts(d_set1):
   for i in d_set1.columns:
       n = d_set1[i].nunique()
       print(i, ": ", n)
unique_counts(d_set1)


# In[14]:


d_set1['TotalPrice'] = d_set1['Quantity'] * d_set1['UnitPrice']


# In[15]:


d_set1['InvoiceDate'].min()


# In[16]:


d_set1['InvoiceDate'].max()


# In[17]:


import datetime as dt
today = dt.datetime(2011,12,10)
d_set1['InvoiceDate'] = pd.to_datetime(d_set1['InvoiceDate'])


# In[18]:


RFM_df = d_set1.groupby('CustomerID').agg({'InvoiceDate': lambda x: (today - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
RFM_df['InvoiceDate'] = RFM_df['InvoiceDate'].astype(int)
RFM_df.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalPrice': 'monetary_value'}, inplace=True)


# In[19]:


quantiles = RFM_df.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[20]:


RFM_seg = RFM_df


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


RFM_seg['r_quartile'] = RFM_seg['recency'].apply(RScore, args=('recency',quantiles,))
RFM_seg['f_quartile'] = RFM_seg['frequency'].apply(FMScore, args=('frequency',quantiles,))
RFM_seg['m_quartile'] = RFM_seg['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))


# In[23]:


RFM_seg.head()


# In[24]:


RFM_seg['RFMScore'] = RFM_seg.r_quartile.map(str) + RFM_seg.f_quartile.map(str) + RFM_seg.m_quartile.map(str)


# In[25]:


RFM_seg.head()


# In[26]:


RFM_seg[RFM_seg['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)

 

# In[27]:


import matplotlib.pyplot as plt
# In[40]:

def RFM_n(RFM):
    return (RFM - RFM.mean())/RFM.std()

# In[28]:


RFM_norm= RFM_df.iloc[:, 0:3]
RFM_norm.recency = RFM_n(RFM_norm.recency)
RFM_norm.frequency = RFM_n(RFM_norm.frequency)
RFM_norm.monetary_value = RFM_n(RFM_norm.monetary_value)

# In[29]:


RFM_norm.head(5)


# In[30]:


# Using the dendrogram helps to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(RFM_norm, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[34]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(RFM_norm)


# In[35]:


RFM_df['clusters'] = y_hc


# In[36]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

colors = ['blue', 'red', 'green', 'red', 'black']

for i in range(0,5):
    ax.scatter(RFM_norm.recency[y_hc == i], RFM_norm.frequency[y_hc == i], RFM_norm.monetary_value[y_hc == i], c = colors[i])

