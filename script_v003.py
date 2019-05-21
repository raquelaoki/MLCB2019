'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
#import matplotlib.pyplot as plt
import sys 
from sklearn.model_selection import train_test_split
import gc
sys.path.append('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\script_v003.p')

#import scrip_v003_def as pgm
from script_v003_def import *

'''
Notes: 
- check if output functions is correct, use a small sample to check if it's saving correctly
- if theta.csv still having problems, i can save only average value for logisct regression 
and deal with this later. 
- make code to do predictions
- memory problems, i might need to save files in the middle and free memory, not sure how
- limite was 920, using gc it was 990. 
'''


'''Important parameters I need to constantly change'''
k = 100
sim = 3000
start_time = time.time()
id = '0001'


'''Loading dataset'''
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final_log.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final_sub.csv"
data = pd.read_csv(filename, sep=',')


'''Splitting Dataset'''
#data = data.iloc[:, 0:10000]
#data = data.sample(n=).reset_index(drop=True)
data, test = train_test_split(data, test_size=0.3, random_state=42)
#print(data.shape, test.shape)


'''Organizing columns names'''
lr = data.columns[[2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y = data.columns[3]
remove = data.columns[[0,1]]
#print(lr,y,remove)


'''
Class to work with model parameters
I thought about using the default values as chain starting values, 
however, i encouter problems to change the size of arrays and matrices 
according with my currently k
'''
class parameters:
    def __init__(self, latent_v,latent_cj,latent_sk,latent_ev,latent_phi ,latent_tht, prediction):
        self.ln = latent_v #array with parameters that are only one number [0-c0,1-gamma0]
        self.la_cj = latent_cj #aaray J
        self.la_sk = latent_sk #array K
        self.la_ev = latent_ev #array V
        self.lm_phi = latent_phi #matrix (jk)
        self.lm_tht = latent_tht #matrix  (kv)      
        self.p = prediction #array [intercept, gender, 15 cancer types, k genes]



'''Non informative prioris: dirichlet has only 1, gamma distribution with 1 average, etc'''
#UPDATE NUMBERS ACCORDING WITH POISSON AND LOG(N)
aux = len(lr)+1
data = data.drop(remove,axis = 1)
v = (data.shape[1]-aux)
j = data.shape[0]
start = parameters([1.65,1.65], #ln [0-c0,1-gamma0]
                   np.repeat(2.72,j), #la_cj
                   np.repeat(2.72,k), #la_sk
                   np.repeat(1,v), #la_ev
                   np.repeat(1/(data.shape[1]-aux),(data.shape[1]-aux)*k).reshape((data.shape[1]-aux),k),#lm_phi v x k 
                   np.repeat(7.42,(data.shape[0])*k).reshape(k,(data.shape[0])), #lm_theta k x j
                   np.concatenate(([-(k*7.42)], np.repeat(1,k+aux-1))))  #p, k+aux-1  because intercept is already counted


start_time = time.time()
output_p, output_f, acept_P,acept_F = MCMC(start,sim,data,k,lr,y)
end_time = time.time() - start_time
np.set_printoptions(threshold=50)

print("--- %s seconds ---" % (time.time() - start_time))


'''WORK IN PROGRESS'''

'''1- saving output'''
#output_part1(output_p,output_f,sim,id)
#output_part2(output_p,output_f,sim,id)
#output_part3(output_p,output_f,sim,id)



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

