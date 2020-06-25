#!/usr/bin/env python
# coding: utf-8

# You can create additional projects and notebooks as you work with Azure Notebooks. You can create notebooks from scratch, or you can upload existing notebooks.
# 
# Jupyter notebooks are highly interactive, and since they can include executable code, they provide the perfect platform for manipulating data and building predictive models from it.
# 
# Enter the following command into the first cell of the notebook:

# In[2]:


get_ipython().system(u'curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv')


# In[3]:


import pandas as pd

df = pd.read_csv('flightdata.csv')
df.head()


# ### Clean and prepare data

# Before you can prepare a dataset, you need to understand its content and structure. In the previous lab, you imported a dataset containing on-time arrival information for a major U.S. airline. That data included 26 columns and thousands of rows, with each row representing one flight and containing information such as the flight's origin, destination, and scheduled departure time. You also loaded the data into a Jupyter notebook and used a simple Python script to create a Pandas DataFrame from it.
# 
# A DataFrame is a two-dimensional labeled data structure. The columns in a DataFrame can be of different types, just like columns in a spreadsheet or database table. It is the most commonly used object in Pandas. In this exercise, you will examine the DataFrame — and the data inside it — more closely.

# In[4]:


df.shape

Getting a row and column count

Now take a moment to examine the 26 columns in the dataset. They contain important information such as the date that the flight took place (YEAR, MONTH, and DAY_OF_MONTH), the origin and destination (ORIGIN and DEST), the scheduled departure and arrival times (CRS_DEP_TIME and CRS_ARR_TIME), the difference between the scheduled arrival time and the actual arrival time in minutes (ARR_DELAY), and whether the flight was late by 15 minutes or more (ARR_DEL15).

Here is a complete list of the columns in the dataset. Times are expressed in 24-hour military time. For example, 1130 equals 11:30 a.m. and 1500 equals 3:00 p.m.

TABLE 1
Column	Description
YEAR	Year that the flight took place
QUARTER	Quarter that the flight took place (1-4)
MONTH	Month that the flight took place (1-12)
DAY_OF_MONTH	Day of the month that the flight took place (1-31)
DAY_OF_WEEK	Day of the week that the flight took place (1=Monday, 2=Tuesday, etc.)
UNIQUE_CARRIER	Airline carrier code (e.g., DL)
TAIL_NUM	Aircraft tail number
FL_NUM	Flight number
ORIGIN_AIRPORT_ID	ID of the airport of origin
ORIGIN	Origin airport code (ATL, DFW, SEA, etc.)
DEST_AIRPORT_ID	ID of the destination airport
DEST	Destination airport code (ATL, DFW, SEA, etc.)
CRS_DEP_TIME	Scheduled departure time
DEP_TIME	Actual departure time
DEP_DELAY	Number of minutes departure was delayed
DEP_DEL15	0=Departure delayed less than 15 minutes, 1=Departure delayed 15 minutes or more
CRS_ARR_TIME	Scheduled arrival time
ARR_TIME	Actual arrival time
ARR_DELAY	Number of minutes flight arrived late
ARR_DEL15	0=Arrived less than 15 minutes late, 1=Arrived 15 minutes or more late
CANCELLED	0=Flight was not cancelled, 1=Flight was cancelled
DIVERTED	0=Flight was not diverted, 1=Flight was diverted
CRS_ELAPSED_TIME	Scheduled flight time in minutes
ACTUAL_ELAPSED_TIME	Actual flight time in minutes
DISTANCE	Distance traveled in miles
The dataset includes a roughly even distribution of dates throughout the year, which is important because a flight out of Minneapolis is less likely to be delayed due to winter storms in July than it is in January. But this dataset is far from being "clean" and ready to use. Let's write some Pandas code to clean it up.

One of the most important aspects of preparing a dataset for use in machine learning is selecting the "feature" columns that are relevant to the outcome you are trying to predict while filtering out columns that do not affect the outcome, could bias it in a negative way, or might produce multicollinearity. Another important task is to eliminate missing values, either by deleting the rows or columns containing them or replacing them with meaningful values. In this exercise, you will eliminate extraneous columns and replace missing values in the remaining columns.
# One of the first things data scientists typically look for in a dataset is missing values. There's an easy way to check for missing values in Pandas. To demonstrate, execute the following code in a cell at the end of the notebook:

# In[5]:


df.isnull().values.any()


# #### Checking for missing values
# ##### The next step is to find out where the missing values are. To do so, execute the following code:

# In[6]:


df.isnull().sum()


# Number of missing values in each column
# 
# Curiously, the 26th column ("Unnamed: 25") contains 11,231 missing values, which equals the number of rows in the dataset. This column was mistakenly created because the CSV file that you imported contains a comma at the end of each line. To eliminate that column, add the following code to the notebook and execute it:

# In[7]:


df = df.drop('Unnamed: 25', axis=1)
df.isnull().sum()


# The DataFrame with column 26 removed
# 
# The DataFrame still contains a lot of missing values, but some of them aren't useful because the columns containing them are not relevant to the model that you are building. The goal of that model is to predict whether a flight you are considering booking is likely to arrive on time. If you know that the flight is likely to be late, you might choose to book another flight.
# 
# The next step, therefore, is to filter the dataset to eliminate columns that aren't relevant to a predictive model. For example, the aircraft's tail number probably has little bearing on whether a flight will arrive on time, and at the time you book a ticket, you have no way of knowing whether a flight will be cancelled, diverted, or delayed. By contrast, the scheduled departure time could have a lot to do with on-time arrivals. Because of the hub-and-spoke system used by most airlines, morning flights tend to be on time more often than afternoon or evening flights. And at some major airports, traffic stacks up during the day, increasing the likelihood that later flights will be delayed.
# 
# Pandas provides an easy way to filter out columns you don't want. Execute the following code in a new cell at the end of the notebook:

# In[8]:


df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
df.isnull().sum()


# The output shows that the DataFrame now includes only the columns that are relevant to the model, and that the number of missing values is greatly reduced:
The only column that now contains missing values is the ARR_DEL15 column, which uses 0s to identify flights that arrived on time and 1s for flights that didn't. Use the following code to show the first five rows with missing values:
# In[9]:


df[df.isnull().values.any(axis=1)].head()


# The reason these rows are missing ARR_DEL15 values is that they all correspond to flights that were canceled or diverted. You could call dropna on the DataFrame to remove these rows. But since a flight that is canceled or diverted to another airport could be considered "late," let's use the fillna method to replace the missing values with 1s.
# 
# Use the following code to replace missing values in the ARR_DEL15 column with 1s and display rows 177 through 184:

# In[10]:


df = df.fillna({'ARR_DEL15': 1})
df.iloc[177:185]


# The dataset is now "clean" in the sense that missing values have been replaced and the list of columns has been narrowed to those most relevant to the model. But you're not finished yet. There is more to do to prepare the dataset for use in machine learning.
# 
# The CRS_DEP_TIME column of the dataset you are using represents scheduled departure times. The granularity of the numbers in this column — it contains more than 500 unique values — could have a negative impact on accuracy in a machine-learning model. This can be resolved using a technique called binning or quantization. What if you divided each number in this column by 100 and rounded down to the nearest integer? 1030 would become 10, 1925 would become 19, and so on, and you would be left with a maximum of 24 discrete values in this column. Intuitively, it makes sense, because it probably doesn't matter much whether a flight leaves at 10:30 a.m. or 10:40 a.m. It matters a great deal whether it leaves at 10:30 a.m. or 5:30 p.m.
# 
# In addition, the dataset's ORIGIN and DEST columns contain airport codes that represent categorical machine-learning values. These columns need to be converted into discrete columns containing indicator variables, sometimes known as "dummy" variables. In other words, the ORIGIN column, which contains five airport codes, needs to be converted into five columns, one per airport, with each column containing 1s and 0s indicating whether a flight originated at the airport that the column represents. The DEST column needs to be handled in a similar manner.

# In this exercise, you will "bin" the departure times in the CRS_DEP_TIME column and use Pandas' get_dummies method to create indicator columns from the ORIGIN and DEST columns.
# 
# Use the following command to display the first five rows of the DataFrame:

# In[11]:


df.head()


# The DataFrame with unbinned departure times
# 
# Use the following statements to bin the departure times:

# In[12]:


import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
df.head()

Confirm that the numbers in the CRS_DEP_TIME column now fall in the range 0 to 23:The DataFrame with binned departure times

Now use the following statements to generate indicator columns from the ORIGIN and DEST columns, while dropping the ORIGIN and DEST columns themselves:
# In[13]:


df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()

The DataFrame with indicator columns

Use the File -&gt; Save and Checkpoint command to save the notebook.

The dataset looks very different than it did at the start, but it is now optimized for use in machine learning.
# ### Build Machine Learning Model

# To create a machine learning model, you need two datasets: one for training and one for testing. In practice, you often have only one dataset, so you split it into two. In this exercise, you will perform an 80-20 split on the DataFrame you prepared in the previous lab so you can use it to train a machine learning model. You will also separate the DataFrame into feature columns and label columns. The former contains the columns used as input to the model (for example, the flight's origin and destination and the scheduled departure time), while the latter contains the column that the model will attempt to predict — in this case, the ARR_DEL15 column, which indicates whether a flight will arrive on time.
# 
# Switch back to the Azure notebook that you created in the previous section. If you closed the notebook, you can sign back into the Microsoft Azure Notebooks portal , open your notebook, and use the Cell -&gt; Run All to rerun the all of the cells in the notebook after opening it.
# 
# In a new cell at the end of the notebook, enter and execute the following statements:

# In[14]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)


# The first statement imports scikit-learn's train_test_split helper function. The second line uses the function to split the DataFrame into a training set containing 80% of the original data, and a test set containing the remaining 20%. The random_state parameter seeds the random-number generator used to do the splitting, while the first and second parameters are DataFrames containing the feature columns and the label column.
# 
# train_test_split returns four DataFrames. Use the following command to display the number of rows and columns in the DataFrame containing the feature columns used for training:

# In[15]:


train_x.shape

Now use this command to display the number of rows and columns in the DataFrame containing the feature columns used for testing:
# In[16]:


test_x.shape

How do the two outputs differ, and why?

Can you predict what you would see if you called shape on the other two DataFrames, train_y and test_y? If you're not sure, try it and find out.
# In[17]:


train_y.shape


# In[18]:


train_x.shape


# There are many types of machine learning models. One of the most common is the regression model, which uses one of a number of regression algorithms to produce a numeric value — for example, a person's age or the probability that a credit-card transaction is fraudulent. You'll train a classification model, which seeks to resolve a set of inputs into one of a set of known outputs. A classic example of a classification model is one that examines e-mails and classifies them as "spam" or "not spam." Your model will be a binary classification model that predicts whether a flight will arrive on-time or late ("binary" because there are only two possible outputs).
# 
# One of the benefits of using scikit-learn is that you don't have to build these models — or implement the algorithms that they use — by hand. Scikit-learn includes a variety of classes for implementing common machine learning models. One of them is RandomForestClassifier, which fits multiple decision trees to the data and uses averaging to boost the overall accuracy and limit overfitting.

# Execute the following code in a new cell to create a RandomForestClassifier object and train it by calling the fit method.

# In[20]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)

# Training the model

Now call the predict method to test the model using the values in test_x, followed by the score method to determine the mean accuracy of the model:
# In[21]:


predicted = model.predict(test_x)
model.score(test_x, test_y)


# Testing the model
# 
# The mean accuracy is 86%, which seems good on the surface. However, mean accuracy isn't always a reliable indicator of the accuracy of a classification model. Let's dig a little deeper and determine how accurate the model really is — that is, how adept it is at determining whether a flight will arrive on time.
# There are several ways to measure the accuracy of a classification model. One of the best overall measures for a binary classification model is Area Under Receiver Operating Characteristic Curve (sometimes referred to as "ROC AUC"), which essentially quantifies how often the model will make a correct prediction regardless of the outcome. In this unit, you'll compute an ROC AUC score for the model you built previously and learn about some of the reasons why that score is lower than the mean accuracy output by the score method. You'll also learn about other ways to gauge the accuracy of the model.
# 
# Before you compute the ROC AUC, you must generate prediction probabilities for the test set. These probabilities are estimates for each of the classes, or answers, the model can predict. For example, [0.88199435, 0.11800565] means that there's an 89% chance that a flight will arrive on time (ARR_DEL15 = 0) and a 12% chance that it won't (ARR_DEL15 = 1). The sum of the two probabilities adds up to 100%.
Run the following code to generate a set of prediction probabilities from the test data:
# In[23]:


from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test_x)

Now use the following statement to generate an ROC AUC score from the probabilities using scikit-learn's roc_auc_score method:
# In[24]:


roc_auc_score(test_y, probabilities[:, 1])


# Generating an AUC score - 67%
# 
# Why is the AUC score lower than the mean accuracy computed in the previous exercise?
# 
# The output from the score method reflects how many of the items in the test set the model predicted correctly. This score is skewed by the fact that the dataset the model was trained and tested with contains many more rows representing on-time arrivals than rows representing late arrivals. Because of this imbalance in the data, you're more likely to be correct if you predict that a flight will be on time than if you predict that a flight will be late.
# 
# ROC AUC takes this into account and provides a more accurate indication of how likely it is that a prediction of on-time or late will be correct.
# 
# You can learn more about the model's behavior by generating a confusion matrix, also known as an error matrix. The confusion matrix quantifies the number of times each answer was classified correctly or incorrectly. Specifically, it quantifies the number of false positives, false negatives, true positives, and true negatives. This is important, because if a binary classification model trained to recognize cats and dogs is tested with a dataset that is 95% dogs, it could score 95% simply by guessing "dog" every time. But if it failed to identify cats at all, it would be of little value.
# 
# Use the following code to produce a confusion matrix for your model:

# In[25]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predicted)


# Generating a confusion matrix
# 
# But look at the second row, which represents flights that were delayed. The first column shows how many delayed flights were incorrectly predicted to be on time. The second column shows how many flights were correctly predicted to be delayed. Clearly, the model isn't nearly as adept at predicting that a flight will be delayed as it is at predicting that a flight will arrive on time. What you want in a confusion matrix is large numbers in the upper-left and lower-right corners, and small numbers (preferably zeros) in the upper-right and lower-left corners.
# 
# Other measures of accuracy for a classification model include precision and recall. Suppose the model was presented with three on-time arrivals and three delayed arrivals, and that it correctly predicted two of the on-time arrivals, but incorrectly predicted that two of the delayed arrivals would be on time. In this case, the precision would be 50% (two of the four flights it classified as being on time actually were on time), while its recall would be 67% (it correctly identified two of the three on-time arrivals). You can learn more about precision and recall from https://en.wikipedia.org/wiki/Precision_and_recall
# 
# Scikit-learn contains a handy method named precision_score for computing precision. To quantify the precision of your model, execute the following statements:

# In[26]:


from sklearn.metrics import precision_score

train_predictions = model.predict(train_x)
precision_score(train_y, train_predictions)

Examine the output. What is your model's precision? -99%Measuring precision

Scikit-learn also contains a method named recall_score for computing recall. To measure you model's recall, execute the following statements:
# In[27]:


from sklearn.metrics import recall_score

recall_score(train_y, train_predictions)

What is the model's recall? - 86%
# Use the File -&gt; Save and Checkpoint command to save the notebook.
# 
# In the real world, a trained data scientist would look for ways to make the model even more accurate. Among other things, they would try different algorithms and take steps to tune the chosen algorithm to find the optimum combination of parameters. Another likely step would be to expand the dataset to millions of rows rather than a few thousand and also attempt to reduce the imbalance between late and on-time arrivals. But for our purposes, the model is fine as-is.

# ### Visualize Output of Model
# 
# In this unit, you'll import Matplotlib into the notebook you've been working with and configure the notebook to support inline Matplotlib output.
# 
# 1. Switch back to the Azure notebook that you created in the previous section. If you closed the notebook, you can sign back into the Microsoft Azure Notebooks portal , open your notebook, and use the Cell -&gt; Run All to rerun the all of the cells in the notebook after opening it.
# 
# 2. Execute the following statements in a new cell at the end of the notebook. Ignore any warning messages that are displayed related to font caching:

# In[28]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# The first statement is one of several magic commands supported by the Python kernel that you selected when you created the notebook. It enables Jupyter to render Matplotlib output in a notebook without making repeated calls to show. And it must appear before any references to Matplotlib itself. The final statement configures Seaborn to enhance the output from Matplotlib.
# 
# 3. To see Matplotlib at work, execute the following code in a new cell to plot the ROC curve for the machine-learning model you built in the previous lab:

# In[29]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# ROC curve generated with Matplotlib
# 
# The dotted line in the middle of the graph represents a 50-50 chance of obtaining a correct answer. The blue curve represents the accuracy of your model. More importantly, the fact that this chart appears at all demonstrates that you can use Matplotlib in a Jupyter notebook.
# 
# The reason you built a machine-learning model is to predict whether a flight will arrive on time or late. In this exercise, you'll write a Python function that calls the machine-learning model you built in the previous lab to compute the likelihood that a flight will be on time. Then you'll use the function to analyze several flights.
# 
# Enter the following function definition in a new cell, and then run the cell.

# In[30]:


def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]


# This function takes as input a date and time, an origin airport code, and a destination airport code, and returns a value between 0.0 and 1.0 indicating the probability that the flight will arrive at its destination on time. It uses the machine-learning model you built in the previous lab to compute the probability. And to call the model, it passes a DataFrame containing the input values to predict_proba. The structure of the DataFrame exactly matches the structure of the DataFrame we used earlier.

# Note: Date input to the predict_delay function use the international date format dd/mm/year.
# 
# 2. Use the code below to compute the probability that a flight from New York to Atlanta on the evening of October 1 will arrive on time. The year you enter is irrelevant because it isn't used by the model.

# In[31]:


predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')


# Predicting whether a flight will arrive on time
# 
# 3. Modify the code to compute the probability that the same flight a day later will arrive on time:

# In[32]:


predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL')


# How likely is this flight to arrive on time? If your travel plans were flexible, would you consider postponing your trip for one day?
# 
# 4. Now modify the code to compute the probability that a morning flight the same day from Atlanta to Seattle will arrive on time:

# In[33]:


predict_delay('2/10/2018 10:00:00', 'ATL', 'SEA')


# Is this flight likely to arrive on time?
# 
# You now have an easy way to predict, with a single line of code, whether a flight is likely to be on time or late. Feel free to experiment with other dates, times, origins, and destinations. But keep in mind that the results are only meaningful for the airport codes ATL, DTW, JFK, MSP, and SEA because those are the only airport codes the model was trained with.
# 
# 4. Execute the following code to plot the probability of on-time arrivals for an evening flight from JFK to ATL over a range of days:

# In[34]:


import numpy as np

labels = ('Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7')
values = (predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('3/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('4/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('5/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('6/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('7/10/2018 21:45:00', 'JFK', 'ATL'))
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))


# Probability of on-time arrivals for a range of dates
# 
# 3. Modify the code to produce a similar chart for flights leaving JFK for MSP at 1:00 p.m. on April 10 through April 16. How does the output compare to the output in the previous step?
# 
# 4. On your own, write code to graph the probability that flights leaving SEA for ATL at 9:00 a.m., noon, 3:00 p.m., 6:00 p.m., and 9:00 p.m. on January 30 will arrive on time. Confirm that the output matches this:
# 
# 

# Probability of on-time arrivals for a range of times
# 
# If you are new to Matplotlib and would like to learn more about it, you will find an excellent tutorial at https://www.labri.fr/perso/nrougier/teaching/matplotlib/. There is much more to Matplotlib than what was shown here, which is one reason why it is so popular in the Python community.





