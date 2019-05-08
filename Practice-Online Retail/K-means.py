
# In[1]:


import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')
d_set = pd.read_excel("Online_Retail.xlsx")
d_set.head()
d_set1 = d_set


# In[3]:


d_set1.head()


# In[4]:


d_set1.Country.nunique()


# In[5]:


d_set1.Country.unique()


# In[6]:


customer_country=d_set1[['Country','CustomerID']].drop_duplicates()
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


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


d_set1 = d_set1[(d_set1['Quantity']>0)]
d_set1.shape
d_set1.info()


# In[14]:


def unique_counts(d_set1):
   for i in d_set1.columns:
       n = d_set1[i].nunique()
       print(i, ": ", n)
unique_counts(d_set1)


# In[15]:


d_set1['TotalPrice'] = d_set1['Quantity'] * d_set1['UnitPrice']


# In[16]:


d_set1['InvoiceDate'].min()


# In[17]:


d_set1['InvoiceDate'].max()


# In[18]:


import datetime as dt
today = dt.datetime(2011,12,10)
d_set1['InvoiceDate'] = pd.to_datetime(d_set1['InvoiceDate'])


# In[19]:


RFM_df = d_set1.groupby('CustomerID').agg({'InvoiceDate': lambda x: (today - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
RFM_df['InvoiceDate'] = RFM_df['InvoiceDate'].astype(int)
RFM_df.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalPrice': 'monetary_value'}, inplace=True)


# In[20]:


RFM_df.head()


# In[21]:


quantiles = RFM_df.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[22]:


RFM_seg = RFM_df


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


RFM_seg['r_quartile'] = RFM_seg['recency'].apply(RScore, args=('recency',quantiles,))
RFM_seg['f_quartile'] = RFM_seg['frequency'].apply(FMScore, args=('frequency',quantiles,))
RFM_seg['m_quartile'] = RFM_seg['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))


# In[25]:


RFM_seg.head()


# In[26]:


RFM_seg['RFMScore'] = RFM_seg.r_quartile.map(str) + RFM_seg.f_quartile.map(str) + RFM_seg.m_quartile.map(str)


# In[27]:


RFM_seg.head()


# In[28]:


RFM_seg[RFM_seg['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)


# ## Clustering

# In[29]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[40]:
#normalization
def RFM_n(RFM):
    return (RFM - RFM.mean())/RFM.std()

# In[41]:
#normalize
RFM_norm= RFM_df.iloc[:, 0:3]
RFM_norm.recency = RFM_n(RFM_norm.recency)
RFM_norm.frequency = RFM_n(RFM_norm.frequency)
RFM_norm.monetary_value = RFM_n(RFM_norm.monetary_value)

# In[41]:


RFM_norm.head(5)


# In[49]:


arr = []
for i in range(1,11):
    K_m = KMeans(n_clusters=i, init='k-means++', random_state=0)
    K_m.fit(RFM_norm)
    arr.append(K_m.inertia_)
    
plt.plot(range(1,11), arr)
plt.title('Elbow graph')
plt.xlabel('Cluster number')
plt.ylabel('WCSS')
plt.show()


# In[47]:


K_m = KMeans(n_clusters=4, init='k-means++', random_state=0)
clusters = K_m.fit_predict(RFM_norm)

RFM_df['clusters'] = clusters
RFM_df.head()
centroids = K_m.cluster_centers_

# In[45]:
#Testing
data_test = pd.DataFrame({'recency': [2.34, 0.38, 2.97, 1.043368, 1.23357], 'frequency': [25.64367, 1.37985, -0.233343, 0.498764, -0.54569],'monetary_value': [6.066890, 7.237305, -0.173343, 0.132313, 1.25632]})
data_test["label"] = K_m.predict(data_test[['recency', 'frequency','monetary_value']].values)
data_test


# In[]:

RFM_df['clusters'] = clusters
RFM_df.head()
# In[48]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
#ax.scatter(RFM_norm.recency, RFM_norm.frequency, RFM_norm.monetary_value, s=30)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
colors = ['blue', 'yellow', 'green', 'red']
for i in range(0,4):
    ax.scatter(RFM_norm.recency[RFM_df.clusters == i], RFM_norm.frequency[RFM_df.clusters == i], RFM_norm.monetary_value[RFM_df.clusters == i], c = colors[i])
    

# In[]:


fig = plt.figure(6)
ax = Axes3D(fig)
ax.set_xlabel('r_quartile')
ax.set_ylabel('f_quartile')
ax.set_zlabel('m_quartile')
ax.scatter(RFM_seg['r_quartile'], RFM_seg['f_quartile'],RFM_seg['m_quartile'], c ='b' )
 