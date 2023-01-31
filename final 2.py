#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 10)


# In[2]:


df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.groupby('area_type')['area_type'].agg('count')


# In[5]:


df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df2.head()


# In[6]:


df2.isnull().sum()


# In[7]:


df3 = df2.dropna()
df3.isnull().sum()


# In[8]:


df3.shape


# In[9]:


df3['size'].unique()


# In[10]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[11]:


df3.head()


# In[12]:


df3 = df3.drop(['size'], axis='columns')
df3.head()


# In[13]:


df3['bhk'].unique()


# In[14]:


df3[df3['bhk']>20]


# In[15]:


df3['total_sqft'].unique()


# In[16]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


df3[~df3['total_sqft'].apply(is_float)].head(10) #range


# In[18]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[19]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num) # Apply the fuction
df4.head()


# In[20]:


df4.loc[30]


# In[21]:


df5 = df4.copy()


# In[22]:


df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft'] #adding new columns
df5.head()


# In[23]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[24]:


len(location_stats[location_stats<=10])


# In[25]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[26]:


len(df5.location.unique())


# In[27]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[28]:


df5.head()


# ## Outliers Removing

# In[29]:


df5.shape


# In[30]:


df6 = df5[~(df5['total_sqft']/df5['bhk']<300)] # Remove entries whose sqft/bhk is less than 300
df6.head()


# In[31]:


df6.shape


# In[32]:


df6.price_per_sqft.describe()


# In[33]:


def remove_pps_outliers(df): # Function to remove outliers for every location
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft) # Calculate mean
        st = np.std(subdf.price_per_sqft) # Calculate SD
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


# In[34]:


df7 = remove_pps_outliers(df6)
df7.shape


# In[35]:


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price per square feet')
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7, "Hebbal")


# In[36]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)
df8.shape


# In[37]:


plot_scatter_chart(df8, "Hebbal")


# In[38]:


df8[df8.bath > df8.bhk+2]


# In[39]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[40]:


df10 = df9.drop(['price_per_sqft'], axis='columns')
df10.head()


# In[41]:


dummies = pd.get_dummies(df10.location)
dummies.head()


# In[42]:


df11 = pd.concat([df10, dummies.drop('other', axis="columns")], axis='columns') # Append df10 and dummies dataframe
df11.head()


# In[43]:


df12 = df11.drop('location', axis='columns')
df12.head()


# In[44]:


df12.shape


# In[45]:


X = df12.drop('price', axis='columns')
X.head()


# In[46]:


y = df12.price
y.head()


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[48]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression() 
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[49]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) # ShuffleSplit will randomize the sample
cross_val_score(LinearRegression(), X,y, cv=cv)


# In[51]:


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return lr_clf.predict([x])[0]


# In[52]:


predict_price('1st Phase JP Nagar', 1000, 2, 2)


# In[ ]:





# In[54]:





# In[55]:





# In[ ]:




