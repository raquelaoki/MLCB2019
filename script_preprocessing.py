#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np

#Load dataset
filename1 = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\tcga_cli.txt"
filename2 = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\tcga_rna.txt"
#filename1 = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\tcga_cli.txt"
#filename2 = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\tcga_rna.txt"
datac = pd.read_csv(filename1, sep=';')
datar = pd.read_csv(filename2, sep=';')

#Checking the size of the datasets
#print('clinical data: ',datac.shape, '\nRNA data:', datar.shape)

#Creating a variable indicator with the prediction value in the 0/1 format
datac['y'] = datac['new_tumor_event_dx_indicator']
datac['y'] = datac['y'].replace(['NO','YES'],[0,1])
datac['gender'] = datac['gender'].replace(['MALE','FEMALE'],[0,1])

#Checking if the values are ok
#print(datac['new_tumor_event_dx_indicator'].value_counts(),'\n', datac['y'].value_counts())

#selecting important clinical features and creating dummy/binary variables for future use
datac_s = datac[['patients','gender','abr','y']]
datac_s = pd.get_dummies(datac_s,columns =['abr'])
#datac_s.head()

#raw counts weren't integer, so I will use a round function to round it. 
datar = datar.drop(columns = 'patients2')
#print(datar.head())
for i in np.arange(1,datar.shape[1]):
    datar.iloc[:,i] = round(datar.iloc[:,i])
    
#print(datar.head())

#Merging two datasets. First columns have the clinical information.
#print('shapes before: ',datac_s.shape, datar.shape)
data = pd.merge(datac_s,datar,on='patients')
#print('shape after', data.shape)

data.to_csv('C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final.csv',sep=',')


# In[ ]:




