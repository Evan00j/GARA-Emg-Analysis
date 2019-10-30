#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from biosppy.signals import emg
from biosppy import plotting
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.cluster import KMeans


# In[5]:


import fileinput
for lines in fileinput.FileInput("NEWrest.txt", inplace=1): 
    lines = lines.strip()
    if lines == '': continue
    print(lines)


# In[6]:


rest = np.loadtxt('NEWrest.txt')


# In[7]:


#Calculate RestAvg
restavg = np.average(rest)
restavg


# In[8]:


import fileinput
for lines in fileinput.FileInput("Newstrong.txt", inplace=1): 
    lines = lines.strip()
    if lines == '': continue
    print(lines)


# In[9]:


forceful = np.loadtxt('NEWstrong.txt')


# In[10]:


forceful.size


# In[11]:


out = emg.emg(signal=forceful, sampling_rate=(forceful.size/30), show=True)


# In[12]:


forceful1, forceful2,forceful3, forceful4 = np.split(forceful[0:9248],4)


# In[13]:


forceful1.size


# In[14]:


out = emg.emg(signal=forceful1, sampling_rate=(forceful.size/30), show=True)


# In[15]:


classvar = np.zeros(len(forceful), dtype=int)
classvar


# In[16]:


df = pd.DataFrame({'data':forceful, 'class':classvar, 'MAV':classvar, 'MAVS':classvar,'SSI':classvar, 'VAR':classvar,'RMS':classvar, 'WL':classvar, 'Trigger':classvar})


# In[17]:


df.loc[12:25,'class'] = 0


# In[18]:


df


# In[19]:


for index, row in df.iterrows():
    if row['data'] > (1.4+restavg):
        df.loc[index, 'class'] = 1


# In[20]:


with pd.option_context("display.max_rows", 1000):
    display(df)


# In[21]:


print(len(df.index))


# In[22]:


#Calculate MAV
for index, row in df.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2 < len(df.index)):
            sum = sum + df.loc[index+index2, 'data']
    if(index+index2 < len(df.index)):
        df.loc[index+index2, 'MAV'] = sum/50


# In[23]:


#Calculate MAVS
for index, row in df.iterrows():
    if(index+1 < len(df.index)):
        mavs = df.loc[index+1, 'MAV'] - df.loc[index, 'MAV']
        df.loc[index, 'MAVS'] = mavs


# In[24]:


#Calculate SSI
for index, row in df.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2 < len(df.index)):
            sum = sum + abs(df.loc[index+index2, 'data'])**2
    if(index+index2 < len(df.index)):
        df.loc[index+index2, 'SSI'] = sum


# In[25]:


#Calculate VAR
for index, row in df.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2 < len(df.index)):
            sum = sum + abs(df.loc[index+index2, 'data'])**2
    if(index+index2 < len(df.index)):
        df.loc[index+index2, 'VAR'] = sum/49


# In[26]:


#Calculate RMS
for index, row in df.iterrows():
    tmp = df.loc[index, 'SSI']/50
    final = np.sqrt(tmp)
    df.loc[index, 'RMS'] = final


# In[27]:


#Calculate WL
for index, row in df.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2+1 < len(df.index)):
            sum = sum + abs((df.loc[index+index2, 'data'] + df.loc[index+index2+1, 'data']))
    if(index+index2 < len(df.index)):
        df.loc[index+index2, 'WL'] = sum/50


# In[28]:


#Calculate Trigger
for index, row in df.iterrows():
    sum = 0
    for index2 in range(10):
        if(index+index2 < len(df.index)):
            sum = sum + df.loc[index+index2, 'class']
    if(index+index2 < len(df.index)):
        if(sum/10 > .75):
            df.loc[index+index2, 'Trigger'] = 1
        else:
            df.loc[index+index2, 'Trigger'] = 0


# In[29]:


#df
with pd.option_context("display.max_rows", 1000):
    display(df)


# In[ ]:





# In[30]:


for lines in fileinput.FileInput("NEWsoft.txt", inplace=1): 
    lines = lines.strip()
    if lines == '': continue
    print(lines)
soft = np.loadtxt('NEWsoft.txt')


# In[31]:


soft.size


# In[32]:


soft1, soft2, soft3, soft4 = np.split(soft[0:9248],4)


# In[33]:


out = emg.emg(signal=soft, sampling_rate=(soft.size/30), show=True)


# In[34]:


out = emg.emg(signal=soft1, sampling_rate=(soft.size/30), show=True)


# In[35]:


classvar = np.zeros(len(soft), dtype=int)
classvar


# In[36]:


dfs = pd.DataFrame({'data':soft, 'class':classvar, 'MAV':classvar, 'MAVS':classvar,'SSI':classvar, 'VAR':classvar,'RMS':classvar, 'WL':classvar, 'Trigger':classvar})


# In[37]:


for index, row in dfs.iterrows():
    if row['data'] > (1.0+restavg):
        dfs.loc[index, 'class'] = 1


# In[38]:


#dfs
#with pd.option_context("display.max_rows", 1000):
#    display(dfs)


# In[39]:


#Calculate MAV
for index, row in dfs.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2 < len(dfs.index)):
            sum = sum + dfs.loc[index+index2, 'data']
    if(index+index2 < len(df.index)):
        dfs.loc[index+index2, 'MAV'] = sum/50


# In[40]:


#Calculate MAVS
for index, row in dfs.iterrows():
    if(index+1 < len(dfs.index)):
        mavs = dfs.loc[index+1, 'MAV'] - dfs.loc[index, 'MAV']
        dfs.loc[index, 'MAVS'] = mavs


# In[41]:


#Calculate SSI
for index, row in dfs.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2 < len(dfs.index)):
            sum = sum + abs(dfs.loc[index+index2, 'data'])**2
    if(index+index2 < len(dfs.index)):
        dfs.loc[index+index2, 'SSI'] = sum


# In[42]:


#Calculate VAR
for index, row in dfs.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2 < len(dfs.index)):
            sum = sum + abs(dfs.loc[index+index2, 'data'])**2
    if(index+index2 < len(dfs.index)):
        dfs.loc[index+index2, 'VAR'] = sum/49


# In[43]:


#Calculate RMS
for index, row in dfs.iterrows():
    tmp = dfs.loc[index, 'SSI']/50
    final = np.sqrt(tmp)
    dfs.loc[index, 'RMS'] = final


# In[44]:


#Calculate WL
for index, row in dfs.iterrows():
    sum = 0
    for index2 in range(50):
        if(index+index2+1 < len(dfs.index)):
            sum = sum + abs((dfs.loc[index+index2, 'data'] + dfs.loc[index+index2+1, 'data']))
    if(index+index2 < len(dfs.index)):
        dfs.loc[index+index2, 'WL'] = sum/50


# In[45]:


#Calculate Trigger
for index, row in dfs.iterrows():
    sum = 0
    for index2 in range(10):
        if(index+index2 < len(dfs.index)):
            sum = sum + dfs.loc[index+index2, 'class']
    if(index+index2 < len(dfs.index)):
        if(sum/10 > .75):
            dfs.loc[index+index2, 'Trigger'] = 2
        else:
            dfs.loc[index+index2, 'Trigger'] = 0


# In[46]:


#dfs
with pd.option_context("display.max_rows", 1000):
    display(dfs)


# In[47]:


#df = df.drop(['class'], axis=1)
#dfs = dfs.drop(['class'], axis=1)
df.dtypes


# In[48]:


train = df.loc[0:len(df.index)*.75]
len(train)


# In[49]:


test = df.loc[len(train.index):len(df.index)]
len(test)


# In[50]:


trains = dfs.loc[0:len(dfs.index)*.75]
len(trains)


# In[51]:


tests = dfs.loc[len(trains.index):len(dfs.index)]
len(tests)


# In[52]:


train = train.append(trains,ignore_index=True)


# In[53]:


test = test.append(tests,ignore_index=True)


# In[54]:


len(train)


# In[55]:


len(test)


# In[56]:


target = train.pop('Trigger')


# In[57]:


target


# In[58]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train, target)


# In[59]:


test = test.dropna()
answers = test.pop("Trigger")
test


# In[60]:


clf.score(test, answers)


# In[61]:


predictions = clf.predict(test)
predictions = np.asarray(predictions)
predictions = pd.DataFrame(predictions)


# In[62]:


answers = pd.DataFrame(answers)


# In[72]:


compare = pd.concat([predictions, answers], axis=1, sort=False)
with pd.option_context("display.max_rows", 4000):
    display(compare)


# In[2]:





# In[64]:


print(dict(zip(train.columns, clf.feature_importances_)))


# In[65]:


from joblib import dump, load
dump(clf, 'expMod.joblib') 
clf2 = load('expMod.joblib') 
clf2.score(test, answers)


# In[66]:


import pickle
filename = 'finalized_modelv2.sav'
pickle.dump(clf,open(filename, 'wb'))
clf3 = pickle.load(open(filename, 'rb'))
clf3.score(test, answers)


# In[67]:


clf2.predict([[0.20,0.0,.268,.00002,2.89,.059,.2407,.480]])


# In[71]:


import struct;
print (struct.calcsize("P") * 8)


# In[68]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(train)


# In[69]:


kmeans

