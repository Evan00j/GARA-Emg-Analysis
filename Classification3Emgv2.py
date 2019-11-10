#!/usr/bin/env python
# coding: utf-8

# In[329]:


import numpy as np
import pandas as pd
from biosppy.signals import emg
from biosppy import plotting
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.cluster import KMeans


# In[330]:


from pathlib import Path

data_folder = Path("FastBaudEMG/")

file_to_open = data_folder / "TripleREST.txt"

f = open(file_to_open)

print(f.read())


# In[331]:


f = open("ARest.txt","w+")
f1 = open("BRest.txt", "w+")
f2 = open("CRest.txt", "w+")
import fileinput
for lines in fileinput.FileInput('FastBaudEMG/TripleREST.txt', inplace=1): 
    lines = lines.strip()
    if lines == '': continue
    if lines[0] == 'a':  f.write(lines[1:] + "\n")
    if lines[0] == 'b':  f1.write(lines[1:] + "\n")
    if lines[0] == 'c':  f2.write(lines[1:] + "\n")
    print(lines)


# In[332]:


Arest = np.loadtxt('ARest.txt')
Brest = np.loadtxt('BRest.txt')
Crest = np.loadtxt('CRest.txt')


# In[333]:


#Calculate RestAvg
Arestavg = np.average(Arest)
Brestavg = np.average(Brest)
Crestavg = np.average(Crest)
Arestavg


# In[334]:


def processText(Aname, Bname, Cname, filein):
    Aname = np.array([0])
    Bname = np.array([0])
    Cname = np.array([0])
    start = 0;
    import fileinput
    for lines in fileinput.FileInput(filein, inplace=1): 
        lines = lines.strip()
        if lines == '': continue
        if lines[0] == 'a': start = 1
        if lines[0] == 'a': Aname = np.append(Aname, [float(lines[1:])])
        if lines[0] == 'b' and start == 1: Bname = np.append(Bname, [float(lines[1:])])
        if lines[0] == 'c'and start == 1: Cname = np.append(Cname, [float(lines[1:])])
        print(lines)
    Aname = Aname[1:]
    Bname = Bname[1:]
    Cname = Cname[1:]
    
    if Bname.size > Cname.size: 
        Bname = Bname[:(Bname.size-1)]
    if Aname.size > Bname.size:
        Aname = Aname[:(Aname.size-1)]
    return Aname, Bname, Cname


# ## Process and Strip Triple Index Hard into ATIH, BTIH and CTIH

# In[335]:


ATIH, BTIH, CTIH = processText(ATIH, BTIH, CTIH, "FastBaudEMG/TripleIndexHard1.txt")


# In[336]:


out = emg.emg(signal=ATIH, sampling_rate=(500), show=True)


# ## Process and Strip Thumb Hard into ATTH, BTTH and CTTH

# In[337]:


ATTH, BTTH, CTTH = processText(ATTH, BTTH, CTTH, "FastBaudEMG/TripleThumbHard2.txt")


# In[338]:


out = emg.emg(signal=ATTH, sampling_rate=(250), show=True)


# # Process and Strip Ring Middle Hard into ATRMH, BTRMH and CTRMH

# In[358]:


ATRMH, BTRMH, CTRMH = processText(ATRMH, BTRMH, CTRMH, "FastBaudEMG/TripleRMHard1.txt")


# In[340]:


out = emg.emg(signal=BTRMH, sampling_rate=(250), show=True)


# # Process and Strip Ring Middle Soft into ATRMS, BTRMS and CTRMS
# ## Use 1 or 3

# In[341]:


ATRMS, BTRMS, CTRMS = processText(ATRMS, BTRMS, CTRMS, "FastBaudEMG/TripleRMSoft2.txt")


# In[342]:


out = emg.emg(signal=BTRMS, sampling_rate=(250), show=True)


# # Process and Strip Open Wrist Flex into AOWF, BOWF and COWF

# In[343]:


ATOWF, BTOWF, CTOWF = processText(ATOWF, BTOWF, CTOWF, "FastBaudEMG/TripleOpenWristFlex2.txt")


# In[344]:


out = emg.emg(signal=CTOWF, sampling_rate=(250), show=True)


# In[345]:


classvar = np.zeros(len(ATIH), dtype=int)
classvar


# In[346]:


CTIH.size


# # Start with Triple Index Hard

# In[347]:


dfTIH = pd.DataFrame({'AData':ATIH, 'BData':BTIH, 'CData':CTIH, 'ATrigger':classvar, 'BTrigger':classvar, 'CTrigger':classvar, 'MAVA':classvar, 'MAVB':classvar, 'MAVC':classvar, 'MAVSA':classvar, 'MAVSB':classvar, 'MAVSC':classvar,'SSIA':classvar, 'SSIB':classvar, 'SSIC':classvar, 'VARA':classvar, 'VARB':classvar, 'VARC':classvar,'RMSA':classvar, 'RMSB':classvar, 'RMSC':classvar, 'WLA':classvar, 'WLB':classvar, 'WLC':classvar, 'Trigger':classvar})


# # We want to determine our Onset on EMG A since it is what triggers the hardest

# In[348]:


for index, row in dfTIH.iterrows():
    if row['AData'] > (.8+Arestavg):
        dfTIH.loc[index, 'ATrigger'] = 1
    if row['BData'] > (.8+Brestavg):
        dfTIH.loc[index, 'BTrigger'] = 1
    if row['CData'] > (.8+Crestavg):
        dfTIH.loc[index, 'CTrigger'] = 1


# In[349]:


def claculateDF(df, MT, Tval):
    #Calculate MAV
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + df.loc[index+index2, 'AData']
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'MAVA'] = sum/50
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + df.loc[index+index2, 'BData']
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'MAVB'] = sum/50
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + df.loc[index+index2, 'CData']
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'MAVC'] = sum/50
        #Calculate MAVS
    for index, row in df.iterrows():
        if(index+1 < len(df.index)):
            mavs = df.loc[index+1, 'MAVA'] - df.loc[index, 'MAVA']
            df.loc[index, 'MAVSA'] = mavs
    for index, row in df.iterrows():
        if(index+1 < len(df.index)):
            mavs = df.loc[index+1, 'MAVB'] - df.loc[index, 'MAVB']
            df.loc[index, 'MAVSB'] = mavs
    for index, row in df.iterrows():
        if(index+1 < len(df.index)):
            mavs = df.loc[index+1, 'MAVC'] - df.loc[index, 'MAVC']
            df.loc[index, 'MAVSC'] = mavs
        #Calculate SSI
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + abs(df.loc[index+index2, 'AData'])**2
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'SSIA'] = sum
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + abs(df.loc[index+index2, 'BData'])**2
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'SSIB'] = sum
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + abs(df.loc[index+index2, 'CData'])**2
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'SSIC'] = sum
                #Calculate VAR
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + abs(df.loc[index+index2, 'AData'])**2
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'VARA'] = sum/49
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + abs(df.loc[index+index2, 'BData'])**2
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'VARB'] = sum/49
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2 < len(df.index)):
                sum = sum + abs(df.loc[index+index2, 'CData'])**2
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'VARC'] = sum/49
                #Calculate RMS
    for index, row in df.iterrows():
        tmp = df.loc[index, 'SSIA']/50
        final = np.sqrt(tmp)
        df.loc[index, 'RMSA'] = final
    for index, row in df.iterrows():
        tmp = df.loc[index, 'SSIB']/50
        final = np.sqrt(tmp)
        df.loc[index, 'RMSB'] = final
    for index, row in df.iterrows():
        tmp = df.loc[index, 'SSIC']/50
        final = np.sqrt(tmp)
        df.loc[index, 'RMSC'] = final
            #Calculate WL
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2+1 < len(df.index)):
                sum = sum + abs((df.loc[index+index2, 'AData'] + df.loc[index+index2+1, 'AData']))
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'WLA'] = sum/50
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2+1 < len(df.index)):
                sum = sum + abs((df.loc[index+index2, 'BData'] + df.loc[index+index2+1, 'BData']))
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'WLB'] = sum/50
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(50):
            if(index+index2+1 < len(df.index)):
                sum = sum + abs((df.loc[index+index2, 'CData'] + df.loc[index+index2+1, 'CData']))
        if(index+index2 < len(df.index)):
            df.loc[index+index2, 'WLC'] = sum/50
                #Calculate Trigger
    for index, row in df.iterrows():
        sum = 0
        for index2 in range(10):
            if(index+index2 < len(df.index)):
                sum = sum + df.loc[index+index2, MT]
        if(index+index2 < len(df.index)):
            if(sum/10 > .75):
                df.loc[index+index2, 'Trigger'] = Tval
            else:
                df.loc[index+index2, 'Trigger'] = 0


# In[350]:


claculateDF(dfTIH, "ATrigger", 1)


# # Now Triple Thumb Hard

# In[351]:


classvar = np.zeros(len(ATTH), dtype=int)
classvar


# In[352]:


dfTTH = pd.DataFrame({'AData':ATTH, 'BData':BTTH, 'CData':CTTH, 'ATrigger':classvar, 'BTrigger':classvar, 'CTrigger':classvar, 'MAVA':classvar, 'MAVB':classvar, 'MAVC':classvar, 'MAVSA':classvar, 'MAVSB':classvar, 'MAVSC':classvar,'SSIA':classvar, 'SSIB':classvar, 'SSIC':classvar, 'VARA':classvar, 'VARB':classvar, 'VARC':classvar,'RMSA':classvar, 'RMSB':classvar, 'RMSC':classvar, 'WLA':classvar, 'WLB':classvar, 'WLC':classvar, 'Trigger':classvar})


# In[353]:


for index, row in dfTTH.iterrows():
    if row['AData'] > (.8+Arestavg):
        dfTTH.loc[index, 'ATrigger'] = 1
    if row['BData'] > (.8+Brestavg):
        dfTTH.loc[index, 'BTrigger'] = 1
    if row['CData'] > (.8+Crestavg):
        dfTTH.loc[index, 'CTrigger'] = 1


# In[354]:


claculateDF(dfTTH, "ATrigger", 2)


# In[355]:


dfTTH.head(1000)


# # Ring Middle Hard

# In[356]:


classvar = np.zeros(len(BTRMH), dtype=int)
classvar


# In[359]:


dfTRMH = pd.DataFrame({'AData':ATRMH, 'BData':BTRMH, 'CData':CTRMH, 'ATrigger':classvar, 'BTrigger':classvar, 'CTrigger':classvar, 'MAVA':classvar, 'MAVB':classvar, 'MAVC':classvar, 'MAVSA':classvar, 'MAVSB':classvar, 'MAVSC':classvar,'SSIA':classvar, 'SSIB':classvar, 'SSIC':classvar, 'VARA':classvar, 'VARB':classvar, 'VARC':classvar,'RMSA':classvar, 'RMSB':classvar, 'RMSC':classvar, 'WLA':classvar, 'WLB':classvar, 'WLC':classvar, 'Trigger':classvar})


# In[360]:


for index, row in dfTRMH.iterrows():
    if row['AData'] > (.8+Arestavg):
        dfTRMH.loc[index, 'ATrigger'] = 1
    if row['BData'] > (.8+Brestavg):
        dfTRMH.loc[index, 'BTrigger'] = 1
    if row['CData'] > (.8+Crestavg):
        dfTRMH.loc[index, 'CTrigger'] = 1


# In[361]:


claculateDF(dfTRMH, "BTrigger", 3)


# In[362]:


dfTRMH.head(1000)


# # Ring Middle Soft

# In[363]:


classvar = np.zeros(len(ATRMS), dtype=int)
classvar


# In[364]:


dfTRMS = pd.DataFrame({'AData':ATRMS, 'BData':BTRMS, 'CData':CTRMS, 'ATrigger':classvar, 'BTrigger':classvar, 'CTrigger':classvar, 'MAVA':classvar, 'MAVB':classvar, 'MAVC':classvar, 'MAVSA':classvar, 'MAVSB':classvar, 'MAVSC':classvar,'SSIA':classvar, 'SSIB':classvar, 'SSIC':classvar, 'VARA':classvar, 'VARB':classvar, 'VARC':classvar,'RMSA':classvar, 'RMSB':classvar, 'RMSC':classvar, 'WLA':classvar, 'WLB':classvar, 'WLC':classvar, 'Trigger':classvar})


# In[365]:


for index, row in dfTRMS.iterrows():
    if row['AData'] > (.8+Arestavg):
        dfTRMS.loc[index, 'ATrigger'] = 1
    if row['BData'] > (.8+Brestavg):
        dfTRMS.loc[index, 'BTrigger'] = 1
    if row['CData'] > (.8+Crestavg):
        dfTRMS.loc[index, 'CTrigger'] = 1


# In[366]:


claculateDF(dfTRMS, "BTrigger", 4)


# In[367]:


dfTRMS.head(1000)


# # Open Wrist Flex

# In[368]:


classvar = np.zeros(len(ATOWF), dtype=int)
classvar


# In[369]:


dfTOWF = pd.DataFrame({'AData':ATOWF, 'BData':BTOWF, 'CData':CTOWF, 'ATrigger':classvar, 'BTrigger':classvar, 'CTrigger':classvar, 'MAVA':classvar, 'MAVB':classvar, 'MAVC':classvar, 'MAVSA':classvar, 'MAVSB':classvar, 'MAVSC':classvar,'SSIA':classvar, 'SSIB':classvar, 'SSIC':classvar, 'VARA':classvar, 'VARB':classvar, 'VARC':classvar,'RMSA':classvar, 'RMSB':classvar, 'RMSC':classvar, 'WLA':classvar, 'WLB':classvar, 'WLC':classvar, 'Trigger':classvar})


# In[370]:


for index, row in dfTOWF.iterrows():
    if row['AData'] > (.8+Arestavg):
        dfTOWF.loc[index, 'ATrigger'] = 1
    if row['BData'] > (.8+Brestavg):
        dfTOWF.loc[index, 'BTrigger'] = 1
    if row['CData'] > (.8+Crestavg):
        dfTOWF.loc[index, 'CTrigger'] = 1


# In[371]:


claculateDF(dfTOWF, "CTrigger", 5)


# In[372]:


dfTOWF.head(1000)


# # Split Apart Train and Test

# In[373]:


def TrainTest(df):
    train = df.loc[0:len(df.index)*.75]
    test = df.loc[len(train.index):len(df.index)]
    return train, test


# In[374]:


trainTIH, testTIH = TrainTest(dfTIH)
trainTTH, testTTH = TrainTest(dfTTH)
trainTRMH, testTRMH = TrainTest(dfTRMH)
trainTRMS, testTRMS = TrainTest(dfTRMS)
trainTOWF, testTOWF = TrainTest(dfTOWF)


# In[375]:


print(dfTIH.index.size, trainTIH.index.size, testTIH.index.size)
print(dfTTH.index.size, trainTTH.index.size, testTTH.index.size)
print(dfTRMH.index.size, trainTRMH.index.size, testTRMH.index.size)
print(dfTRMS.index.size, trainTRMS.index.size, testTRMS.index.size)
print(dfTOWF.index.size, trainTOWF.index.size, testTOWF.index.size)


# In[376]:


traintot = pd.concat([trainTIH, trainTOWF, trainTRMH, trainTRMS, trainTTH])
traintot.index.size


# In[377]:


testtot = pd.concat([testTIH, testTOWF, testTRMH, testTRMS, testTTH])
testtot.index.size


# In[378]:


target = traintot.pop('Trigger')


# In[379]:


target


# In[380]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(traintot, target)


# In[381]:


answers = testtot.pop("Trigger")


# In[382]:


clf.score(testtot, answers)


# In[383]:


predictions = clf.predict(testtot)
predictions = np.asarray(predictions)
predictions = pd.DataFrame(predictions)


# In[384]:


answersdf = pd.DataFrame(answers)
answersdf = answersdf.reset_index()
del answersdf['index']


# In[385]:


compare = pd.concat([predictions, answersdf], axis=1, sort=False)
with pd.option_context("display.max_rows", None):
    display(compare)


# In[386]:


print(dict(zip(traintot.columns, clf.feature_importances_)))


# In[387]:


from joblib import dump, load
dump(clf, '3emg.joblib') 
clf2 = load('3emg.joblib') 
clf2.score(testtot, answers)


# In[ ]:





# In[ ]:





# In[ ]:




