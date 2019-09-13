# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:51:04 2019

@author: raoki
"""

'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
from sklearn.model_selection import train_test_split
import sys 
import matplotlib.pyplot as plt
from scipy.stats import norm
import pymc3 as pm


'''
Notes:
Check p2_meeting_v007 for theory

ToDo: change website


'''


'''Hyperparameters'''
k = 100
sim = 2000
bach_size = 500
step1 = 10
step2 = 20
id = '01'

if bach_size//step2 <= 20:
    print('ERROR ON MCMC, this division must be bigger than 20')

'''Loading dataset'''
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final_log.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final_sub.csv"
data = pd.read_csv(filename, sep=',')


'''Splitting Dataset'''
#data = data.iloc[:, 0:1000]
#data = data.sample(n=1000).reset_index(drop=True)
data, test = train_test_split(data, test_size=0.3, random_state=42)

'''Organizing columns names'''
lr = data.columns[[2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y = data.columns[3]
remove = data.columns[[0,1]]


'''Runnning in batches and saving the partial outputs in files'''
start_time = time.time()


'''USING PYMC3'''
#Cj is constant 121
theta_cov = np.zeros(shape=(k,k))
np.fill_diagonal(theta_cov,121)


train = data.drop(lr, axis = 1)
train = train.drop(remove,axis = 1)
y01 = data[y]
train = train.drop(y, axis = 1)
train = train.iloc[:,1:1000]

train  = train .values.astype(int)
y01 = y01.as_matrix().reshape(len(y01),1)
v = train.shape[1]
j = train.shape[0]
ab = np.power(v*7.42,4)

print('starting model')
with pm.Model() as model:
        # Priori Distributions 
        gamma0 = pm.Gamma('gamma0',alpha = ab,beta = ab) #one number 
        c0 = pm.Gamma('c0',alpha = ab, beta = ab) #one number
    
        #gamma_sk = pm.Gamma('gamma_sk',  alpha = gamma0, beta = c0)
        #sk = MyDistr('sk', gamma_sk=gamma_sk, shape=k)    
        
        sk = pm.Gamma('sk', alpha = gamma0, beta = c0, shape = k) #k-array
        ev = pm.Gamma('ev',alpha = 1, beta = 1, shape = v) #v-array

        theta = pm.MvNormal('theta', mu=sk,cov = theta_cov,shape=(j,k)) #matrix jxk
        phi = pm.Dirichlet('phi', a = ev, shape = (k,v)) #matrix vxk
        #Likelihood 
        #mu = pm.dot(theta,phi)
        njv = pm.Normal('nvj', mu=pm.math.matrix_dot(theta,phi), observed=train)
        trace = pm.sample(1000, tune=500)
        #trace = pm.sample(1000,tune = 500, init = 'advi', nuts_kwargs={"target_accept":0.9,"max_treedepth": 15})





#pm.traceplot(trace1, var_names=['gamma0', 'c0']);






print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s min ---" % int((time.time() - start_time)/60))
print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))


