#import packages: pandas, numpy, pyplot, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

#use glob to generate list of arrival detail files
arr_stat_list = glob.glob("Detailed*.csv")
arr_stat_list

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

#concatenate list into single df, reset index, drop original index, rename cols, 
#and make the carrier and origin columns categoricals
arr_data = pd.concat(df_list)
arr_data.reset_index(inplace = True)
arr_data = arr_data.drop("index", axis = 1)
arr_data.columns = ['scheduled_arr', 'carrier', 'origin', 'scheduled_elapsed', 'actual_elapsed', 'arr_delay']
arr_data[['carrier', 'origin']] = arr_data[['carrier', 'origin']].astype('category')


#read in airport distance info
distance = pd.read_csv('SLC_routes.csv')
arr_data_dist = arr_data.merge(distance, how = 'left', left_on = 'origin', right_on = 'faa_code')
arr_data_dist.drop('faa_code', axis = 1)


#add cols: 'arr_DateHour', 'date', 'day_name', and 'day_of_year' and 'sceduled_hour'
arr_data_dist['arr_DateHour'] = arr_data_dist['scheduled_arr'].dt.round('H')
arr_data_dist['date'] = arr_data_dist['scheduled_arr'].dt.date
arr_data_dist['day_name'] = arr_data_dist['scheduled_arr'].dt.day_name()
arr_data_dist['day_of_year'] = arr_data_dist['scheduled_arr'].dt.dayofyear
arr_data_dist['date'] = pd.to_datetime(arr_data_dist['date'])
arr_data_dist['scheduled_hour'] = arr_data_dist['scheduled_arr'].dt.hour
arr_data_dist['year'] = arr_data_dist['scheduled_arr'].dt.year



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


#read in weather data
weather = pd.read_csv('kslc_daily_weather_data.csv')
weather["date"] = pd.to_datetime(weather["date"])
weather.drop('wind_fastest_1min', axis = 1, inplace = True)

#fillna on the various missing values with reasonable substitutes
weather["avg_wind"].fillna(weather["avg_wind"].mean(), inplace = True)
weather['water_equiv_on_grd'].fillna(0, inplace = True)
weather['snowfall'].fillna(0, inplace = True)
weather["tavg"].fillna(((weather.tmax + weather.tmin) / 2), inplace = True)


#merge arr_data_dist and precip_hour on 'arr_DateHour' and 'DateHour'
arr_precip_merge = arr_data_dist.merge(precip_hour, how = 'left', left_on = 'arr_DateHour', right_on = 'DateHour')

#merge arr_precip_merge and weather on 'date'
SLC_arrival_merge = arr_precip_merge.merge(weather, how = 'left', on = 'date')

SLC_arrival_merge.columns

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


#fillna the HourPrecip col with the mean for the column and create a column for "ontime"
SLC_arr_ml_cat['HourPrecip'].fillna(SLC_arr_ml_cat['HourPrecip'].mean(), inplace = True)

SLC_arr_ml_cat.info()


SLC_arr_ml_cat['ontime'] = (SLC_arr_ml_cat['arr_delay'] <= 0)

SLC_arr_ml_cat.head()


SLC_arr_ml_cat.to_csv("KSLC_arrivals_tidy.csv")



#read in KSCL arrivals data
df = pd.read_csv("KSLC_arrivals_tidy.csv")

#drop the old index
df.drop("Unnamed: 0", axis = 1, inplace = True)


#create separate dfs with the numeric and categorical variables and scale the numeric df
num_cols = ['scheduled_elapsed', 'distance', 'year','day_of_year', 
                 'scheduled_hour', 'HourPrecip',
                 'snowfall', 'water_equiv_on_grd']
cat_cols = ['carrier', 'day_name', 'ontime']

df_numeric = df[num_cols]
df_cat = df[cat_cols]

npa_num_scaled = scale(df_numeric)
df_num_scaled = pd.DataFrame(npa_num_scaled)
df_num_scaled.columns = num_cols

#concat the numeric and categorical dfs
df_scaled = pd.concat([df_num_scaled, df_cat], axis = 1)


#create dummy variables for "carrier" and "day_name"
df_dummies = pd.get_dummies(df_scaled, drop_first = True)


# In[ ]:


#randomly sample 400k flights from df_dummies
df_dummies_sample = df_dummies.sample(500000, random_state = 21)


#separate out the features and target
X = df_dummies_sample.drop('ontime', axis = 1)
y = df_dummies_sample['ontime']


#perfrom PCA on the dataset
pca = PCA(n_components = 8)
pca.fit(df_dummies_sample)
X_transformed = pca.transform(df_dummies_sample)
print(X_transformed.shape)

#instantiate a LogisticRegression classifier called logreg
logreg = LogisticRegression()


#create a train and test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size = 0.2, random_state = 42)

#create a fit based on the training data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

conf = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(logreg, X_test, y_test, display_labels = ['late', 'on time'], normalize = 'true', cmap = 'Blues')
plt.savefig('cm.png', dpi = 300)

print(classification_report(y_test, y_pred))

#copute predicted probabilities
y_pred_prob = logreg.predict_proba(X_test)[:,1]

#get roc-auc
roc_auc = roc_auc_score(y_test, y_pred_prob)

#create annotation for plot
annot = 'Area Under Curve: {:.3f}'.format(roc_auc)

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.annotate(annot, (0.5, 0.25), c = "DarkRed")
plt.show()