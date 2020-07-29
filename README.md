# Predicting Arrival On-time Status for SLC International Airport
## Supervised Machine Learning

#### Motivation
Late flight arrivals are a hassle, not only for travelers, but also for the people waiting for them at their destination. Fortunately, the [US Bureau of Transportation Statistics](https://www.bts.gov/) has information about flight arrival and departure statistics going back to the 80s. I combined this with distance information obtained from [OpenFlights](https://openflights.org/), and weather information from [NOAA](https://www.ncdc.noaa.gov/cdo-web/search) to create a set of inputs that would help predict whether or not a flight would arrive on time.

#### Methods
I cleaned and merged the data in Python, using the Pandas package. I broke the datetime information into its constituent parts, to create a more robust predictive method. 

I then scaled this cleaned data, and one-hot encoded the categorical variables (e.g. airline and day of the week). Since many of the variables are dependent on one another, I used partial component analysis to strip the data down into its 8 most fundamental components. I applied several different fit methods (decision tree, random forest, and naïve Bayes, among others), but logisitc regression was by far the best. 

#### Outcomes
For flights that were on time, the model predicted with a precision of 73% and a recall of 76%. For late flights, precision was 69% and recall was 65%. The area under the ROC curve was 0.773, indicating a model that has strong predictive power.

This model could be used to help travelers choose appropriate layover times, or to help airlines adjust scheduling and timing.
