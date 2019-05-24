'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
#import matplotlib.pyplot as plt
import sys 
from sklearn.model_selection import train_test_split
sys.path.append('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019')
from script_v004_def import *
'''
Notes: 

- make code to do predictions
- plots
- save a summary
- smaller steps of bach size
- for tht and phi i can save just the sum 
'''


'''Important parameters I need to constantly change'''
k = 100
sim = 4000
bach_size = 500
step1 = 10
step2 = 20
start_time = time.time()
id = '0003'


'''Loading dataset'''
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final_log.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final_sub.csv"
data = pd.read_csv(filename, sep=',')


'''Splitting Dataset'''
#data = data.iloc[:, 0:1000]
#data = data.sample(n=1000).reset_index(drop=True)
data, test = train_test_split(data, test_size=0.3, random_state=42)
#print(data.shape, test.shape)


'''Organizing columns names'''
lr = data.columns[[2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y = data.columns[3]
remove = data.columns[[0,1]]


'''Non informative prioris: dirichlet has only 1, gamma distribution with 1 average, etc'''
#UPDATE NUMBERS ACCORDING WITH POISSON AND LOG(N)
aux = len(lr)+1
data = data.drop(remove,axis = 1)
v = (data.shape[1]-aux)
j = data.shape[0]
start = parameters(np.repeat(1.65,2),#ln [0-c0,1-gamma0]
                   np.repeat(2.72,j), #la_cj
                   np.repeat(2.72,k), #la_sk
                   np.repeat(1,v), #la_ev
                   np.repeat(1/(data.shape[1]-aux),(data.shape[1]-aux)*k).reshape((data.shape[1]-aux),k),#lm_phi v x k 
                   np.repeat(7.42,(data.shape[0])*k).reshape(k,(data.shape[0])), #lm_theta k x j
                   np.concatenate(([-(k*7.42)], np.repeat(1,k+aux-1))))  #p, k+aux-1  because intercept is already counted

'''Runnning in batches and saving the partial outputs in files'''
start_time = time.time()

chain_p = np.tile(start.p.tolist(),(int(bach_size/step1),1))
chain_ln = np.tile(start.ln.tolist(),(int(bach_size/step1),1))
chain_la_sk = np.tile(start.la_sk.tolist(),(int(bach_size/step1),1))
chain_la_cj = np.tile(start.la_cj.tolist(),(int(bach_size/step1),1))
chain_la_ev = np.tile(start.la_ev.tolist(),(int(bach_size/step1),1))
chain_lm_tht = np.tile(start.lm_tht.reshape(-1,1),(1,int(bach_size/step2)))
chain_lm_phi = np.tile(start.lm_phi.reshape(-1,1),(1,int(bach_size/step2)))


for ite in np.arange(0,sim//bach_size):    
    print('iteration--',ite,' of ',sim//bach_size)          
    current, a_P, a_F = MCMC(start,bach_size,data,k,lr,y,id,ite,step1,step2,
                             chain_p,chain_ln,chain_la_sk,chain_la_cj,
                             chain_la_ev,chain_lm_tht,chain_lm_phi)
    start = current

    
end_time = time.time() - start_time
print("--- %s seconds ---" % (time.time() - start_time))

#current, a_P, a_F = MCMC(start,bach_size,data,k,lr,y,chain,id,0)
'''WORK IN PROGRESS'''





