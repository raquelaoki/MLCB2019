'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
#import matplotlib.pyplot as plt
import sys 
from sklearn.model_selection import train_test_split
import gc
sys.path.append('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019')
from pympler import muppy, summary


#import scrip_v003_def as pgm
#from script_v003_def import *
from script_v004_def import *
'''
Notes: 

- make code to do predictions
- alocate in the beggining and save in each iteration 
- save every 10 steps 
- don't save matrix or save every 100 steps


'''


'''Important parameters I need to constantly change'''
k = 100
sim = 400
bach_size = 50
step1 = 1
step2 = 2
start_time = time.time()
id = '0003'


'''Loading dataset'''
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final_log.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final_sub.csv"
data = pd.read_csv(filename, sep=',')


'''Splitting Dataset'''
data = data.iloc[:, 0:1000]
data = data.sample(n=1000).reset_index(drop=True)
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
chain_lm_phi = np.tile(start.lm_tht.reshape(-1,1),(1,int(bach_size/step2)))


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



'''2 - Accuracy  Traning set '''
#Depends on save output2 from previous problem
#df = data[lr]
#df = pd.concat([df, pd.DataFrame()], axis=1)
#print(df.shape)

'''Plots'''
#output_factor_la_sk.shape
#plt.plot(np.arange(0,sim),output_factor_la_sk[:,0], 'r-', alpha=1)
#plt.xlabel('sk_0')
#plt.show()
#print(output_factor_la_sk[:,12])


#plt.plot(np.arange(0,len(output_logistic[:,0])),output_logistic[:,0], 'r-', alpha=1)
#plt.xlabel('Logistic Regression - intercept')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_intercept.png')

#plt.plot(np.arange(0,len(output_logistic[:,1])),output_logistic[:,1], 'r-', alpha=1)
##plt.xlabel('Logistic Regression - gender')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_gender.png')

#plt.plot(np.arange(0,len(output_logistic[:,10])),output_logistic[:,10], 'r-', alpha=1)
#plt.xlabel('Logistic Regression - cancer type (one of them)')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_cancertype.png')

#plt.plot(np.arange(0,len(output_logistic[:,30])),output_logistic[:,30], 'r-', alpha=1)
#plt.xlabel('Logistic Regression - k (one of them)')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_k.png')

#all_objects = muppy.get_objects()
#sum1 = summary.summarize(all_objects)
# Prints out a summary of the large objects
#summary.print_(sum1)
# Get references to certain types of objects such as dataframe
#dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
#for d in dataframes:
 # print(d.columns.values)
 # print(len(d))

#crash kernel
#all_objects = muppy.get_objects()
#sum1 = summary.summarize(all_objects)
#summary.print_(sum1)                          
 
 
