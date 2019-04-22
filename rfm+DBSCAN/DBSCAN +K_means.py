#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("data_ysm.xlsx")
print(df.head())


# In[32]:


df1 = df


# In[33]:


df1.TRANS_ID.nunique()


# In[34]:


df1.isnull().sum(axis=0)


# In[35]:


df1.describe()


# In[36]:


df1.head()


# In[37]:


import datetime as dt
NOW = dt.datetime(2018,10,24)


# In[38]:


df1['OP_DATE_PRECISE'] = pd.to_datetime(df['OP_DATE_PRECISE'], format="%d/%m/%Y")
# Since recency is calculated for a point in time. The last invoice date is 2018-10-24, this is the date we will use to calculate recency.


# In[39]:


df1['OP_DATE_PRECISE'].min()


# In[40]:


df1['OP_DATE_PRECISE'].max()


# In[41]:


# Create a RFM table
rfmTable = df1.groupby('CARD_HOLDER').agg({'OP_DATE_PRECISE': lambda x: (NOW - x.max()).days, # Recency
                                        'TRANS_ID': lambda x: x.count(),      # Frequency
                                        'AMOUNT': lambda x: x.sum()}) # Monetary Value

rfmTable['OP_DATE_PRECISE'] = rfmTable['OP_DATE_PRECISE'].astype(int)
rfmTable.rename(columns={'OP_DATE_PRECISE': 'recency', 
                         'TRANS_ID': 'frequency', 
                         'AMOUNT': 'monetary_value'}, inplace=True)


# In[42]:


rfmTable.head()


# In[43]:


quantiles = rfmTable.quantile(q=[0.20,0.40,0.60,0.80])
quantiles


# In[44]:


quantiles = quantiles.to_dict()
quantiles


# In[45]:


segmented_rfm = rfmTable


# In[46]:


# Lowest recency, highest frequency and monetary are our best customers 
def RScore(x,p,d):
    if x <= d[p][0.20]:
        return 1
    elif x <= d[p][0.40]:
        return 2
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]: 
        return 4
    else:
        return 5
    
def FMScore(x,p,d):
    if x <= d[p][0.20]:
        return 5
    elif x <= d[p][0.40]:
        return 4
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]: 
        return 2
    else:
        return 1


# In[47]:


segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
# Add segment numbers to the RFM table


# In[48]:


segmented_rfm.head()


# In[49]:


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
segmented_rfm.head()


# In[50]:


# Here is top 10 of our best customers!
segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)


# In[51]:


rfm_scores = segmented_rfm.iloc[:,3:6]


# In[53]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_rfm = rfmTable.iloc[:, 0:3]
normalized_rfm = scaler.fit_transform(rfmTable.iloc[:, 0:3])
normalized_rfm = pd.DataFrame(normalized_rfm)
normalized_rfm.columns = ['recency', 'frequency', 'monetary_value']
# cluster the data into five clusters
dbscan = DBSCAN(eps=0.8, min_samples = 5)
clusters = dbscan.fit_predict(normalized_rfm)


# In[54]:


rfmTable['clusters'] = clusters
rfmTable.head()


# In[55]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)


ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

colors = ['blue', 'yellow', 'green', 'red', 'black']
for i in range(-1,4):
    ax.scatter(normalized_rfm.recency[(rfmTable.clusters == i).values], normalized_rfm.frequency[(rfmTable.clusters == i).values], normalized_rfm.monetary_value[(rfmTable.clusters == i).values], c = colors[i])


# In[56]:


extreme_rfmTable = rfmTable[rfmTable.clusters == -1]
scaler = StandardScaler()
extreme_normalized_rfm = extreme_rfmTable.iloc[:, 0:3]
extreme_normalized_rfm = scaler.fit_transform(extreme_rfmTable.iloc[:, 0:3])
extreme_normalized_rfm = pd.DataFrame(extreme_normalized_rfm)
extreme_normalized_rfm.columns = ['recency', 'frequency', 'monetary_value']


# In[57]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
extreme_clusters = kmeans.fit_predict(extreme_normalized_rfm)

extreme_rfmTable['clusters'] = extreme_clusters


# In[58]:


extreme_rfmTable.clusters[extreme_rfmTable.clusters == 0] = 2
extreme_rfmTable.clusters[extreme_rfmTable.clusters == 1] = 3


# In[59]:


fig1 = plt.figure(3)
ax1 = fig1.add_subplot(111, projection='3d')
ax1 = Axes3D(fig1)

#ax.scatter(normalized_rfm.recency, normalized_rfm.frequency, normalized_rfm.monetary_value, s=30)

ax1.set_xlabel('Recency')
ax1.set_ylabel('Frequency')
ax1.set_zlabel('Monetary')

colors1 = ['green', 'red']

for i in range(0,2):
    ax1.scatter(extreme_normalized_rfm.recency[(extreme_rfmTable.clusters == i + 2).values], extreme_normalized_rfm.frequency[(extreme_rfmTable.clusters == i + 2).values], extreme_normalized_rfm.monetary_value[(extreme_rfmTable.clusters == i + 2).values], c = colors1[i])
    


# In[60]:


rfmTable = rfmTable[rfmTable.clusters != -1]
rfmTable = rfmTable.append(extreme_rfmTable)
scaler = StandardScaler()
normalized_rfm = rfmTable.iloc[:, 0:3]
normalized_rfm = scaler.fit_transform(rfmTable.iloc[:, 0:3])
normalized_rfm = pd.DataFrame(normalized_rfm)
normalized_rfm.columns = ['recency', 'frequency', 'monetary_value']


# In[61]:


fig2 = plt.figure(4)
ax2 = fig2.add_subplot(111, projection='3d')
ax2 = Axes3D(fig2)

#ax.scatter(normalized_rfm.recency, normalized_rfm.frequency, normalized_rfm.monetary_value, s=30)

ax2.set_xlabel('Recency')
ax2.set_ylabel('Frequency')
ax2.set_zlabel('Monetary')

colors2 = ['blue', 'yellow', 'green', 'red', 'black']

for i in range(0,4):
    ax2.scatter(normalized_rfm.recency[(rfmTable.clusters == i).values], normalized_rfm.frequency[(rfmTable.clusters == i).values], normalized_rfm.monetary_value[(rfmTable.clusters == i).values], c = colors2[i])
    


# In[ ]:




