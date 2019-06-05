'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys 
sys.path.append('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019')
from script_v005_def import *
import matplotlib.pyplot as plt

'''
Notes: 
- Change my dirichelt, check on overleaf the theory
- https://hackernoon.com/implementation-of-gaussian-naive-bayes-in-python-from-scratch-c4ea64e3944d

'''


'''Important parameters I need to constantly change'''
k = 100
sim = 1000
bach_size = 500
step1 = 10
step2 = 20
id = '0009'

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
                   np.repeat(2.72,k*2).reshape(2,k), #la_sk
                   np.repeat(1.0004,v), #la_ev
                   np.repeat(1/(data.shape[1]-aux),(data.shape[1]-aux)*k).reshape((data.shape[1]-aux),k),#lm_phi v x k 
                   np.repeat(7.42,(data.shape[0])*k).reshape(k,(data.shape[0]))) #lm_theta k x j
                   #np.concatenate(([4], np.repeat(0,k+aux-1))))  #p, k+aux-1  because intercept is already counted

'''Runnning in batches and saving the partial outputs in files'''
start_time = time.time()

#chain_p = np.tile(start.p.tolist(),(int(bach_size/step1),1))
chain_ln = np.tile(start.ln.tolist(),(int(bach_size/step1),1))
chain_la_sk = np.tile(start.la_sk.reshape(-1,1),(1,int(bach_size/step1)))
chain_la_cj = np.tile(start.la_cj.tolist(),(int(bach_size/step1),1))
chain_la_ev = np.tile(start.la_ev.tolist(),(int(bach_size/step1),1))
chain_lm_tht = np.tile(start.lm_tht.reshape(-1,1),(1,int(bach_size/step2)))
chain_lm_phi = np.tile(start.lm_phi.reshape(-1,1),(1,int(bach_size/step2)))


for ite in np.arange(0,sim//bach_size):    
    print('iteration--',ite,' of ',sim//bach_size)   
    #.print('it should be 981',data.shape)       
    current, a_P, a_F = MCMC(start,bach_size,data,k,lr,y,id,ite,step1,step2,
                             chain_ln,chain_la_sk,chain_la_cj,
                             chain_la_ev,chain_lm_tht,chain_lm_phi)
    start = current

    
end_time = time.time() - start_time
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s min ---" % int((time.time() - start_time)/60))
print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))

#current, a_P, a_F = MCMC(start,bach_size,data,k,lr,y,chain,id,0)
'''WORK IN PROGRESS'''

accuracy(sim//bach_size,id,data,j,k)
#accuracy(sim//bach_size,id,test,j,k)
#similarity between patients 
accuracy_final(sim//bach_size,id,data,j,k)



#lask, lacj, laev, ln, p

conv_plots(sim,bach_size,'lask',id)
conv_plots(sim,bach_size,'lacj',id)
conv_plots(sim,bach_size,'laev',id)
conv_plots(sim,bach_size,'ln',id)
conv_plots(sim,bach_size,'p',id)


'''
files = []
for ite in range(sim//bach_size):
    files.append('Data\\output_'+parameter+'_id'+id+'_bach'+str(ite)+'.txt')

f=pd.read_csv(files[0],sep=',', header=None)
for i in range(1,len(files)):
    f = pd.concat([f,pd.read_csv(files[i],sep=',', header=None)],axis =0,sort=False)



if parameter!='ln':
    k = random.sample(range(f.shape[1]),3)
else:
    k = [0,1]

fig, pltarray = plt.subplots(len(k), sharex=True)
pltarray[0].set_title(parameter)   
for i in range(len(k)):
    pltarray[i].plot(np.arange(0,f.shape[0]),f.iloc[:,k[i]], 'r-', alpha=1)
    pltarray[i].set_ylabel('Position '+str(k[i]))

fig.subplots_adjust(hspace=0.3)
plt.show()
'''