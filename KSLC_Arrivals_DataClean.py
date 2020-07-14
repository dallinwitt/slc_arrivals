#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages: pandas, numpy, pyplot, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 


# # Arrival Data

# In[2]:


#use glob to generate list of arrival detail files
arr_stat_list = glob.glob("Detailed*.csv")
arr_stat_list


# In[3]:


#read in CSVs into dummy list
cols = [0, 1, 4, 5, 7, 8, 9]
df_list = []

for i in range(7):
    file = arr_stat_list[i]
    df_list.append(pd.read_csv(file, 
                engine = 'python',
                usecols = cols, 
                dtype = {'Origin Airport':'category'},
                skiprows = 7, 
                parse_dates = [[1, 3]],
                skipfooter = 1
                ))


# In[4]:


#concatenate list into single df, reset index, drop original index, rename cols, 
#and make the carrier and origin columns categoricals
arr_data = pd.concat(df_list)
arr_data.reset_index(inplace = True)
arr_data = arr_data.drop("index", axis = 1)
arr_data.columns = ['scheduled_arr', 'carrier', 'origin', 'scheduled_elapsed', 'actual_elapsed', 'arr_delay']
arr_data[['carrier', 'origin']] = arr_data[['carrier', 'origin']].astype('category')


# In[5]:


#read in airport distance info
distance = pd.read_csv('SLC_routes.csv')
arr_data_dist = arr_data.merge(distance, how = 'left', left_on = 'origin', right_on = 'faa_code')
arr_data_dist.drop('faa_code', axis = 1)


# In[6]:


#add cols: 'arr_DateHour', 'date', 'day_name', and 'day_of_year' and 'sceduled_hour'
arr_data_dist['arr_DateHour'] = arr_data_dist['scheduled_arr'].dt.round('H')
arr_data_dist['date'] = arr_data_dist['scheduled_arr'].dt.date
arr_data_dist['day_name'] = arr_data_dist['scheduled_arr'].dt.day_name()
arr_data_dist['day_of_year'] = arr_data_dist['scheduled_arr'].dt.dayofyear
arr_data_dist['date'] = pd.to_datetime(arr_data_dist['date'])
arr_data_dist['scheduled_hour'] = arr_data_dist['scheduled_arr'].dt.hour
arr_data_dist['year'] = arr_data_dist['scheduled_arr'].dt.year


# # Precipitation Data

# In[7]:


#read in precip data
precip = pd.read_csv('kslc_precip_data.csv',
                    usecols = [2, 3])
precip["DATE"] = pd.to_datetime(precip["DATE"])

#create a dt index, resample the data to hourly, and replace NaN with 0
precip.set_index('DATE', inplace = True)
precip_hour = precip.resample('H').asfreq()
precip_hour.fillna(0, inplace = True)
precip_hour.reset_index(inplace = True)
precip_hour.columns = ['DateHour', 'HourPrecip']


# # Weather Data

# In[8]:


#read in weather data
weather = pd.read_csv('kslc_daily_weather_data.csv')
weather["date"] = pd.to_datetime(weather["date"])
weather.drop('wind_fastest_1min', axis = 1, inplace = True)

#fillna on the various missing values with reasonable substitutes
weather["avg_wind"].fillna(weather["avg_wind"].mean(), inplace = True)
weather['water_equiv_on_grd'].fillna(0, inplace = True)
weather['snowfall'].fillna(0, inplace = True)
weather["tavg"].fillna(((weather.tmax + weather.tmin) / 2), inplace = True)


# # Merge Data

# In[9]:


#merge arr_data_dist and precip_hour on 'arr_DateHour' and 'DateHour'
arr_precip_merge = arr_data_dist.merge(precip_hour, how = 'left', left_on = 'arr_DateHour', right_on = 'DateHour')


# In[10]:


#merge arr_precip_merge and weather on 'date'
SLC_arrival_merge = arr_precip_merge.merge(weather, how = 'left', on = 'date')


# In[12]:


SLC_arrival_merge.columns


# In[13]:


#make a new df with only the info that will be used in the ML model, but keep categoricals for now
SLC_arr_ml_cat = SLC_arrival_merge[['carrier', 
                                    'scheduled_elapsed', 
                                    'distance', 
                                    'year',
                                    'day_name', 
                                    'day_of_year', 
                                    'scheduled_hour', 
                                    'HourPrecip', 
                                    'avg_wind',
                                    'precip', 
                                    'snowfall',
                                    'tavg',
                                    'tmax',
                                    'tmin', 
                                    'water_equiv_on_grd',
                                    'arr_delay']]


# In[ ]:


#fillna the HourPrecip col with the mean for the column and create a column for "ontime"
SLC_arr_ml_cat['HourPrecip'].fillna(SLC_arr_ml_cat['HourPrecip'].mean(), inplace = True)


# In[15]:


SLC_arr_ml_cat.info()


# In[ ]:


SLC_arr_ml_cat['ontime'] = (SLC_arr_ml_cat['arr_delay'] <= 0)


# In[18]:


SLC_arr_ml_cat.head()


# In[19]:


SLC_arr_ml_cat.to_csv("KSLC_arrivals_tidy.csv")


# In[ ]:




