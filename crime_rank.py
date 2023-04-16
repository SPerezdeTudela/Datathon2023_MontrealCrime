#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#HANDLE THE CRIME DATA CSV FILE


# In[4]:


df = pd.read_csv('actes-criminels.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df['CATEGORIE'].value_counts()


# In[7]:


#drop any NaN PDQ rows because our model will take that as input
#Limitations: we lost some crime category data
df = df.dropna(subset=['PDQ'])
df['CATEGORIE'].value_counts()


# In[8]:


#visualize correlations
sns.pairplot(df)


# In[9]:


#we want a multiclass regression, so we labeled
#each of the categories in a new target column
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['CATEGORIE'])
#making sure the numbers match up
print(df['CATEGORIE'].value_counts())
print(df['target'].value_counts())


# In[10]:


#based on the counts, we can build a dict to store which label
#corresponds to which category
target_dict = {0:'Infractions entrainant la mort', 1:'Introduction', 2:'Méfait', 3:'Vol dans / sur véhicule à moteur', 4:'Vol de véhicule à moteur', 5:'Vols qualifiés'}


# In[11]:


#HANDLE LOCATION DATA GEOJSON FILE


# In[12]:


gf = gpd.read_file('limitespdq.geojson')
gf.head()


# In[13]:


gf.info()


# In[14]:


#use the bounds property to get the longitude, latitude ranges each PDQ covers
gf['geometry'].bounds[:5]


# In[15]:


#TRAINING THE MODEL


# In[16]:


#Using a random forest classifer for our model that will
#predict the relative liklihood of each type of crime happening given a location
X_train, X_test, y_train, y_test = train_test_split(df[['PDQ']], df['target'], test_size = 0.2)
model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[17]:


#IMPORTANT: As you can see, our model's score is pretty low.
#Due to the lack of time, we were not able to opitmize its performance.
#However, note that we are doing a relative ranking of the crimes, from likiest to lease likiest.
#So this is not invalidate our model, for we are comparing the class probabilites to each other.


# In[18]:


#THE PROGRAM USING NOMINATIM GEO API
#1. USER INPUTS AN ADDRESS
#2. THE ADDRESS IS CONVERTED TO LONG/LAT COORDINATES
#3. USING THE GEOJSON FILE, FIND THE PDQ THAT THE COORDINATED FALL IN
#4. THE PDQ WILL GO INTO OUR MODEL, AND A RANKING OF THE TYPE OF CRIMES WILL BE PRINTED
#(FROM MOST LIKELY TO LEAST LIKELY)


# In[27]:


import json
import urllib.parse
import urllib.request


# In[35]:


NOMINATIM_BASE = 'https://nominatim.openstreetmap.org'
REFERER = 'azraz'

def _geocoding_url(address):
    '''
    Return the appropriate url to request geocoding information from Nominatim API.
    '''
    query_params = [
        ('q', address),
        ('format', 'json')
        ]

    encoded_params = urllib.parse.urlencode(query_params)
    
    url = f'{NOMINATIM_BASE}/search?{encoded_params}'
    return url

def _download_geocoding(url):
    '''
    Send the a request to the search Nominatim API to get back a json response.
    Return the list of dictionary.
    Handle any errors that may arise in the process.
    '''
    response = None

    try:
        request = urllib.request.Request(url, headers = {'Referer': REFERER})
        response = urllib.request.urlopen(request)
        response_json = response.read().decode(encoding = 'utf-8')
        return json.loads(response_json)
    
    finally:
        if response != None:
            response.close()
            
def _get_center_coordinates():
    '''
    Get the lat,lon of the center from Nominatim's response.
    Handle error that may occur.
    '''
    address = input()
    url = _geocoding_url(address)
    data = _download_geocoding(url)

    center_lat = data[0]['lat']
    center_long = data[0]['lon']

    return float(center_lat), float(center_long)

def _translate_french(word):
    if word == 'Introduction':
        return 'Break in'
    if word == 'Vol dans / sur véhicule à moteur':
        return 'Theft in/on motor vehicle'
    if word == 'Méfait':
        return 'Mischief'
    if word == 'Vol de véhicule à moteur':
        return 'Motor vehicle theft'
    if word == 'Vols qualifiés':
        return 'Robbery'
    if word == 'Infractions entrainant la mort':
        return 'Offenses resulting in death'


# In[36]:


def crime_ranks():
    '''
    Print the ranking of each type of crime, from highest threat to lowest.
    '''
    y_coor, x_coor = _get_center_coordinates()
    target_row = gf.loc[((gf['geometry'].bounds['minx'] <= x_coor) & (gf['geometry'].bounds['maxx'] >= x_coor) & (gf['geometry'].bounds['miny'] <= y_coor) & (gf['geometry'].bounds['maxy'] <= y_coor))]
    target_pdq = target_row['PDQ'].to_frame()
    
    ranks = model.predict_proba(target_pdq)
    l1 = list(enumerate(ranks[0]))
    ranked = sorted(l1, key = lambda x: x[1], reverse = True)
    
    for i in range(len(ranked)):
        category = target_dict[ranked[i][0]]
        print(f'Threat #{i+1}: {_translate_french(category)}')


# In[37]:


crime_ranks()


# In[ ]:


#END OF DATATHON

