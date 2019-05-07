#!/usr/bin/env python
# coding: utf-8

# ## Find Your Best Customers with Customer Segmentation in Python

# ### Imports

# In[1]:


import pandas as pd
import warnings
import datetime as dt


# ### Load data

# In[2]:


warnings.filterwarnings('ignore')
df = pd.read_excel("Online_Retail.xlsx")
df.head()
df1 = df


# ### Preprocessing

# In[3]:


df1.Country.nunique() #38 unique contry name
customer_country=df1[['Country','CustomerID']].drop_duplicates()#fonly distinct values
#finding the most used Country name
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[4]:


#More than 90% of the customers in the data are from the United Kingdom.
#There’s some research indicating that customer clusters vary by geography,
#so here I’ll restrict the data to the United Kingdom only.
df1 = df1.loc[df1['Country'] == 'United Kingdom']
df1.isnull().sum(axis=0) #there exist 133600 missing value on customerId column


# In[5]:


#lest to remove them 
df1 = df1[pd.notnull(df1['CustomerID'])] #removed null values
#Check the minimum values in UnitPrice and Quantity columns.
df1.Quantity.min()
#Remove the negative values in Quantity column.
df1 = df1[(df1['Quantity']>0)]
df1.shape
df1.info()


# ### Check unique value for each column.

# In[6]:


def unique_counts(df1):
   for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)
unique_counts(df1)


# In[7]:


#Add a column for total price.
df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']
#Find out the first and last order dates in the data.
df1['InvoiceDate'].min()#first
df1['InvoiceDate'].max()#last
#Since recency is calculated for a point in time, and the last invoice date is 2011–12–09,
# we will use 2011–12–10 to calculate recency.
NOW = dt.datetime(2011,12,10)
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])


# ### Create a RFM table

# In[8]:


rfmTable = df1.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalPrice': lambda x: x.sum()})
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalPrice': 'monetary_value'}, inplace=True)


# In[9]:


#Calculate RFM metrics for each customer
rfmTable.head()


# ### Split the metrics
# 

# In[10]:


quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()

#Create a segmented RFM table
segmented_rfm = rfmTable


# ### The lowest recency, highest frequency and monetary amounts are our best customers.
# 

# In[11]:


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


# In[12]:


segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
segmented_rfm.head()


# In[13]:


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
segmented_rfm.head()


# In[14]:


segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)


# In[ ]:




