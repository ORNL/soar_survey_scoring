#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:36:44 2021

@author: 2he
"""
#HodgeRank

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy 
import pandas
from  matplotlib import pyplot
import seaborn as sns
sns.set(style='ticks')
plt.style.use('seaborn-whitegrid')


df = pd.read_csv("UserResponses.csv")
UserChoice = pd.read_csv("UserCapabilitiesRanked.csv") 

df = df.drop(["Title"], axis = 1)
df = df.drop(["Survey"], axis = 1)
df = df.drop(["Q1Answer"], axis = 1)
df = df.drop(["Q2Answer"], axis = 1)
df = df.drop(["Q10Answer"], axis = 1)
df = df.drop(["SurveyStatus"], axis = 1)

##Needs improvement, doesn't take row into account
dftot=df.groupby(['SurveyUser']).sum().sum(axis=1) ##Switch to average? Or divided by max?
df.sum(axis=1)


df['User_Weight'] = " "

n = len(dftot)

for i in range (0, n):
    index = df.loc[df['SurveyUser'] == dftot.index[i]]
    df['User_Weight'][index.index] = dftot[i]
##This assumes all users review same number.

df['toolsum'] = df.sum(axis=1)
df["Norm Score"] = df['toolsum']/df['User_Weight'] 

#MAssumes all tools are scored 3 times
dfscore=df.groupby(['Tool']).sum()/3

plotarray = df[["Tool", "Q3Answer"]]
plotarray.columns = ['Tool', 'Answer']
Q = pd.DataFrame(np.repeat(1, len(df)))
plotarray["Question"] =  Q

for i in [2,3,4,5,6, 8]:
    plotarray1 = df[["Tool", list(df.columns)[i]]]
    plotarray1.columns = ['Tool', 'Answer']
    plotarray1["Question"] = pd.DataFrame(np.repeat(i, len(df)))
    plotarray = plotarray.append(plotarray1)
    
fg1 = sns.swarmplot('Question', 'Answer', data = plotarray, hue='Tool')
