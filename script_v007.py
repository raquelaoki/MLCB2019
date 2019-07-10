'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
from sklearn.model_selection import train_test_split
import sys 
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma 
import pymc3 as pm
import copy

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
data_sim = copy.deepcopy(data)
v = data_sim.drop([lr,y,remove], axis = 1)



'''Parameters'''
class parameters:
    __slots__ = ('ln', 'la_cj','la_sk','la_c1','la_ev','lm_phi','lm_tht')   
    def __init__(self, latent_v,latent_cj,latent_sk,latent_c1, latent_ev,latent_phi ,latent_tht):
        self.ln = latent_v #array with parameters that are only one number [0-c0,1-gamma0]
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #string of array K
        self.la_c1 = latent_c1 #string of array V
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format      
        #self.p = prediction #string of array [intercept, gender, 15 cancer types, k genes]



'''Gibbs Sampling'''


def gibbs(current):
    new = copy.deepcopy(current)
    #1: n
    #2: 
    #c1|eta ~ InvGamma() and c1~InvGamma(a0,b0)
    gamma1 = 1
    a0 = 1
    b0 = 1
    new.la_c1 = 1/np.random.gamma(shape = gamma1+a0,scale = 1/current.la_ev+b0), size = len(current.la_ev))
    #eta|c1 ~ Gamma(gamma1,c1)
    new.eta = np.random.gamma(shape = gamma1, scale = new.la_c1, size = len(new.la_c1))
    #phi|eta ~ Dir (eta)
    new.lm_phi = np.random.dirichlet(alpha = new.eta,size = k)    
    
    

'''Runnning in batches and saving the partial outputs in files'''
start_time = time.time()



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




#pm.traceplot(trace1, var_names=['gamma0', 'c0']);






print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s min ---" % int((time.time() - start_time)/60))
print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))


