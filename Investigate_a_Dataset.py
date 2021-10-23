#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset - No-show appointments
# 
# ## Table of Contents
# <ul>
# <li><a href="#intoduction">introduction</a></li>   
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# ## Introduction
# The data set used is the one for showing up for appointments at a hospital, most of the columns show a specific condition with values of zeros and ones to indicate wether they have it or not,
# also data -like when they schedueled their appointment and when it's supposed to be held- is available, some more information about their names, appointment ID, Age, their area of residence,...etc

# <a id='intro'></a>
# 
# 

# Importing the packages that might be needed for analysis

# In[62]:


import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# ### General Properties
# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.

# Loading the data and printing the first few lines of it:

# In[93]:


df= pd.read_csv("noshowappointments-kagglev2-may-2016.csv")
df.head()


# 
# ### Data Cleaning 

# Checking for any issues with the data that might trouble the analysis

# In[64]:


df.info()


# In[33]:


df.duplicated().sum()


# There are no missing or duplicated data but:
# a) the ScheduleDay and AppointmentDay columns are of Datatype string instead of datetime which is eaither to work with
# b) The "No-show" column needs a change in name, values and datatype

# Changing the values of the "No-show" column

# In[94]:


df['No-show'].replace({"No":"1","Yes":"0"}, inplace=True)


# Changing the datatype of the "No-show" column to int

# In[95]:


df['No-show']=df['No-show'].astype(int)


# Renaming the column to something more straight forward

# In[96]:


df.rename(columns={'No-show':'Show'},inplace=True)


# Creating masks for showing-up and not showing up for an easy use

# In[97]:


show_up=df.Show == True
no_show_up=df.Show== False


# Changing the datatype of the ScheduleDay and AppointmentDay columns to datetime 

# In[98]:


df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])


# Creating new columns with extracted data from the scheduleday and the appointment day that might be useful

# In[99]:


df['Sc_Hour']=df['ScheduledDay'].dt.hour
df['Sc_Day']=df['ScheduledDay'].dt.day
df['Sc_Month']=df['ScheduledDay'].dt.month
df['Sc_Year']=df['ScheduledDay'].dt.year
df['App_Day']=df['AppointmentDay'].dt.day
df['App_Month']=df['AppointmentDay'].dt.month
df['App_Year']=df['AppointmentDay'].dt.year


# Droping columns that are no longer useful

# In[100]:


columns_to_drop=['PatientId','AppointmentID','ScheduledDay','AppointmentDay']
df.drop(columns_to_drop, axis=1, inplace=True)


# In[102]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# 
# ### Research Question 1 Does a certain feature like Alcoholism or Diabetes influence their chances of showing-up

# In[112]:


def figure_labels(x,y,titles):
    return plt.xlabel('{}'.format(x)), plt.ylabel('{}'.format(y)) ,plt.title('{}'.format(titles))


# Taking a look at the entire set of data

# In[44]:


df.hist(figsize=(20,16));


# Checking how having a scholarship affects their expectency to showing-up

# In[117]:


df.Scholarship[show_up].hist(alpha=0.7,label='Showing up')
df.Scholarship[no_show_up].hist(alpha=0.7, label='No_showing up');
plt.legend();
figure_labels('Having a scholarship','Showing-up','Scholarship-Showing-up graph');


# Checking which neibourhood has a higher or lower chance of showing up

# In[161]:


df.groupby('Neighbourhood').Show.mean().plot(kind='bar', figsize=(20,8))
figure_labels('Neighbourhoods','expectency to show up', 'Neibourhood-showing-up distribution');


# it seems like the "Parque Mosocoso" is the most showing-up neighbourhood, while "Santos Dumont" is the least one

# In[73]:


df.groupby('Scholarship').Show.mean()


# It seems that having a scholarship slightly affect their showing-up chance negatively

# Checking how alcoholism would make a difference

# In[103]:


df.Alcoholism[show_up].mean(), df.Alcoholism[no_show_up].mean()


# In[18]:


df.groupby('Alcoholism').Show.mean()


# In[118]:


df.Alcoholism[show_up].hist(alpha=0.7,label='Showing up')
df.Alcoholism[no_show_up].hist(alpha=0.7, label='No_showing up');
plt.legend();
figure_labels('Suffering Alcoholism','Showing-up','Alcoholism-Showing-up graph');


# It seems like alcoholism has nothing to do with their showing-up percentage

# Checking how Diabetes would make a difference

# In[20]:


df.groupby('Diabetes').Show.mean()


# In[119]:


df.Diabetes[show_up].hist(alpha=0.7,label='Showing up')
df.Diabetes[no_show_up].hist(alpha=0.7, label='No_showing up');
plt.legend();
figure_labels('Having Diabetes','Showing-up','Diabetes-Showing-up graph');


# Diabetes seems to have a very slight correlation to showing-up (positively)

# Checking how Handcap would make a difference

# In[22]:


df.groupby('Handcap').Show.mean()


# In[129]:


df.groupby('Handcap').Show.mean().plot(kind='bar')
figure_labels('Handcap significance','Showing-up','Handcap-Showing-up mean graph');


# In[130]:


df.Handcap[show_up].hist(alpha=0.7,label='Showing up')
df.Handcap[no_show_up].hist(alpha=0.7, label='No_showing up');
plt.legend();
figure_labels('having a Handcap','Showing-up','Handcap-Showing-up graph');


# It seems like that the higher the level of being handcap the higher the chance of not showing up

# Checking how Hipertension would make a difference

# In[47]:


df.groupby('Hipertension').Show.mean()


# In[131]:


df.Hipertension[show_up].hist(alpha=0.7,label='Showing up')
df.Hipertension[no_show_up].hist(alpha=0.7, label='No_showing up');
plt.legend();
figure_labels('Suffering Hipertension','Showing-up','Hipertension-Showing-up graph');


# It seems like having hipertension slightly affect the propability of showing up (positively)

# Checking how receiving an SMS would make a difference

# In[49]:


df.groupby('SMS_received').Show.mean()


# In[132]:


df.SMS_received[show_up].hist(alpha=0.7,label='Showing up')
df.SMS_received[no_show_up].hist(alpha=0.7, label='No_showing up');
plt.legend();
figure_labels('Receiving an SMS','Showing-up','SMS-Showing-up graph');


# It seems like receiving an SMS is more correlated with not showing up than showing up

# Checking how age would make a difference

# In[166]:


df.Age[show_up].hist(alpha=0.7,label='showing up')
df.Age[no_show_up].hist(alpha=0.7,label='not showing up');
plt.legend()
figure_labels('Age distripution','Showing-up','Age-Showing-up graph');


# Assuming that those under the age of zero are unborn-children, so this means that pregnant women never cease to show up for a check, other than that it doesn't seem to have a real correlation except for maybe around 10-30 have higher chance for not showing up, as well as really old ones

# In[134]:


df.plot(x='Age',y='Show', kind='scatter');
figure_labels('Age','Showing-up','Correlation between age and showing up propability');


# Checking how Gender would make a difference

# In[165]:


df.Gender[show_up].hist(alpha=0.7,label='showing up')
df.Gender[no_show_up].hist(alpha=0.7,label='not showing up')
plt.legend()
figure_labels('Gender','Showing-up','Gender-Showing-up association');


# In[136]:


df.groupby('Gender').Show.mean().plot(kind='bar');
figure_labels('Gender','Showing-up','Gender-Showing-up mean graph');


# apparently more men tend to show up for their appointment

# Checking again how their gender is associated with the charactrestics that has a little bit of higher chance of not showing up

# In[75]:


df.query('Gender=="F"').Scholarship.mean(), df.query('Gender=="M"').Scholarship.mean()


# In[76]:


df.query('Gender=="F"').Hipertension.mean(), df.query('Gender=="M"').Hipertension.mean()


# In[109]:


df.groupby('Gender').Handcap.mean()


# In[78]:


df.query('Gender=="F"').Diabetes.mean(), df.query('Gender=="M"').Diabetes.mean()


# It seems that more females have scholarships than males, but other than that it kinda appears females and males have a fair amount of the charactristics that plays a role in showing up or not

# ### Research Question 2  Is the timing of the appointment and the scheduled day associated with the chances of showing up 

# Checking the distribution of the rush hours and the expectency to showing-up

# In[167]:


df.Sc_Hour[show_up].hist(alpha=0.7,label='showing up')
df.Sc_Hour[no_show_up].hist(alpha=0.7,label='not showing up')
plt.legend()
figure_labels('Scheduled hour','Showing-up','Scheduled_Hour and showing-up');


# There are higher chances of showing up between 6-7, less likely between 10:30-12 and 13:30-15 and the least after 19

# Checking the distribution of the rush days of the month and the expectency to showing-up

# In[170]:


df.Sc_Day[show_up].hist(alpha=0.7,label='showing up')
df.Sc_Day[no_show_up].hist(alpha=0.7,label='not showing up')
plt.legend()
figure_labels('Scheduled Day','Showing-up','scheduled-Day-showing-up graph');


# Checking the distribution of the rush appointment day and the expectency to showing-up

# In[171]:


df.App_Day[show_up].hist(alpha=0.7,label='showing up')
df.App_Day[no_show_up].hist(alpha=0.7,label='not showing up')
plt.legend()
figure_labels('Appointment Day of the month','Showing-up','Appointment-day vs showing_up probability');


# There doesn't seam to be a very strong correlation except for at end of the month (starting on 25) the chances of showing up seem to decline

# Checking the distribution of the rush scheduled months and the expectency to showing-up

# In[172]:


df.Sc_Month[show_up].hist(alpha=0.7,label='showing up')
df.Sc_Month[no_show_up].hist(alpha=0.7,label='not showing up')
plt.legend()
figure_labels('Scheduled month','Showing-up','scheduled_month-Showing-up association');


# Checking the distribution of the rush appointment months and the expectency to showing-up

# In[174]:


df.App_Month[show_up].hist(alpha=0.7,label='showing up')
df.App_Month[no_show_up].hist(alpha=0.7,label='notshowing up')
plt.legend()
figure_labels('Appointment month','Showing-up','appointment month vs showing-up correlation');


# In[143]:


df.groupby('App_Month').Show.mean().plot(kind='bar')
figure_labels('Appointment month number','Showing-up','appointment month vs showing-up mean');


# It seems like those whosw appointments are in the month number 5 have fewer chances to show up

# <a id='conclusions'></a>
# ## Conclusions
# 
# >**Findings summary**:
# 1-The ones with a scholarship tend to skip the appointment
# 2-The higher the level of the Handicap the higher the chance of not showing-up
# 3-people with Hipertension or diabetes show-up more often
# 4-also surprisingly receiving an sms decreases the chances of showing-up
# 5-very old people are less likely to show-up followed by around the age of 20
# 
# > **Limitation**: 
# 1- not having enough data over the financial state of these people, their severity of issue to show-up for or having dependentsor care givers
# 
# > **Further areas of research**:
# getting to know more about the distribution of the neibourhoods, their being stay-at home or having an outside job, having dependents or not
# 
# 

# In[163]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




