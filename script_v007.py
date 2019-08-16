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
- Implemeting Gibbs Sampling 
- Updated the dataset     

'''


'''Hyperparameters'''
k = 100 #Latents Dimension 
sim = 300 #Simulations 
bach_size = 100 #Batch size for memory purposes 
step1 = 5 #Saving chain every step1 steps 
step2 = 10
id = '01' #identification of simulation 

#WRONG< UPDATE HERE 
if bach_size//step2 <= 20:
    print('ERROR ON MCMC, this division must be bigger than 20')

'''Loading dataset'''
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_filtered.txt"
data = pd.read_csv(filename, sep=';')


'''Splitting Dataset'''
train, test = train_test_split(data, test_size=0.3, random_state=22)

'''Organizing columns names'''
remove = train.columns[[0,1,2]]
y = train.columns[2]
y01 = np.array(train[y])
train = train.drop(remove, axis = 1)
y01_t = np.array(test[y])
test = test.drop(remove, axis = 1)


'''Defining variables'''
v = train.shape[1] #genes
j = train.shape[0] #patients 


'''Parameters'''
class parameters:
    __slots__ = ('ln', 'la_cj','la_sk','la_c1','la_ev','lm_phi','lm_tht')   
    def __init__(self, latent_v,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.ln = latent_v #array with parameters that are only one number [0-c0,1-gamma0]
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format      

#need to update these values considering the new dataset 
current = parameters(np.repeat(1.65,2),#ln [0-c0,1-gamma0]
                   np.repeat(2.23,j), #la_cj
                   np.repeat(2.23,k*2).reshape(2,k), #la_sk
                   np.repeat(1.0004,v), #la_ev
                   np.repeat(1/v,v*k).reshape(v,k),#lm_phi v x k 
                   np.repeat(5,j*k).reshape(j,k)) #lm_theta k x j

'''Gibbs Sampling'''
train0 = np.matrix(train)

def gibbs(current,train0,j,v,k,y01):
    new = copy.deepcopy(current) 
    #1: P(l_vjk^t|theta^{t−1},phi^{t−1})
    lvk = np.zeros((v,k))
    ljk = np.zeros((j,k))
    #qk= np.repeat(0.01,k)
    
    for ki in np.arange(k):
        ldotdotk = np.multiply(np.array(current.lm_tht[:,ki].reshape(j,1)),current.lm_phi[:,ki].reshape(1,v))
        ldotdotk = np.multiply(ldotdotk,train0).sum()
        lvk[:,ki] = np.random.multinomial(ldotdotk,current.lm_phi[:,ki])
        ljk[:,ki] = np.random.poisson(current.lm_tht[:,ki])
        #print("--- %s seconds ---" % (time.time() - start_time1))  #about 160s-s80s
        #2: P(lv.k^t|theta^{t−1},phi^{t−1}) 
        #3: P(l.jk^k|theta^{t−1},phi^{t−1})
        #4: P(phi^t|,eta^{t-1})
        #5: P(theta^t|c_j^{t-1},s_k^{t-1})
        #ljk = ljvk.sum(axis = 1)
        #lvk = ljvk.sum(axis = 0)
    
        #lvdk = np.random.multinomial(n = ,pvals = size=1)
        new.lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+current.la_ev),size = 1)
        new.lm_tht[:,ki] = np.random.gamma(shape=(current.la_sk[y01,ki]+ljk[:,ki]),scale=current.la_cj-1)
        
        #Future use
        #qk[ki] = np.random.beta(ldotdotk,current.la_ev.sum()*v)
    #6: P(c_j^t|theta^t,skm^{t-1})
    a1 = 2
    b1 = 2.23
    
    new.la_cj = 1/np.random.gamma(shape = (current.la_sk.sum(axis = 1)[y01]+a1), scale = 1/(b1+new.lm_tht.sum(axis=1)))
    
    #MUST CHECK THE FOLLOWING DISTRIBUTIONS IN THE FUTURE
    #7: P(eta^t|phi^t)
    g1 = 50
    c1 = 50
    #8) P(s_km^t|theta^t,c_j^t)    
    a2 = 1.5
    b2 = 1.5
    uvk  = np.repeat(0,v)
    
    for ki in np.arange(k):
        
        for vi in np.arange(v):
            p = current.la_ev[vi]/(current.la_ev[vi]+np.arange(max(lvk[vi,ki],1))+1)
            uvk[vi] = uvk[vi]+ np.random.binomial(n=1,p=p).sum() 
            
            
         #skj = current.la_sk[y01,ki]
        uk = np.array([0,0])
        for ji in np.arange(j):
            p = current.la_sk[y01[ji],ki]/(current.la_sk[y01[ji],ki]+np.arange(max(ljk[ji,ki],1))+1)
            uk[y01[ji]] = uk[y01[ji]] +np.random.binomial(1,p=p).sum()
            #uk[y01[ji],ki] = uk[y01[ji],ki]+np.random.binomial(1,p=p).sum()
        new.la_sk[:,ki] = 1/np.random.gamma(a2+uk,1/(b2+v))     
    
    #qk1 = np.log(1-qk)
    #qk1 = qk1.sum()
    new.la_ev = np.random.gamma(shape = (g1+uvk),scale = 1/(c1+50))
    return(new)

start_time = time.time()    


new  = gibbs(current,train0,j,v,k,y01)
print(current.lm_tht.mean(),new.lm_tht.mean())
print(current.lm_phi.mean(),new.lm_phi.mean())
print(current.la_sk.mean(),new.la_sk.mean())
print(current.la_cj.mean(),new.la_cj.mean())
print(current.la_ev.mean(),new.la_ev.mean())
current= copy.deepcopy(new )


print("--- %s seconds ---" % (time.time() - start_time))   


'''Creating the chains'''
chain_la_sk = np.tile(current.la_sk.reshape(-1,1),(1,int(bach_size/step1)))
chain_la_cj = np.tile(current.la_cj.tolist(),(int(bach_size/step1),1))
chain_la_ev = np.tile(current.la_ev.tolist(),(int(bach_size/step1),1))
chain_lm_tht = np.tile(current.lm_tht.reshape(-1,1),(1,int(bach_size/step2)))
chain_lm_phi = np.tile(current.lm_phi.reshape(-1,1),(1,int(bach_size/step2)))

'''Starting chain and parametrs'''

start_time = time.time()    
    
for ite in np.arange(0,sim//bach_size):    
    count_s1 = 0
    count_s2 = 0
    chain_la_sk[:,count_s1]=current.la_sk.reshape(1,-1)
    chain_la_cj[count_s1]=current.la_cj.tolist()
    chain_la_ev[count_s1]=current.la_ev.tolist()
    chain_lm_tht[:,count_s2]=current.lm_tht.reshape(1,-1)
    chain_lm_phi[:,count_s2]=current.lm_phi.reshape(1,-1)
    print('iteration--',ite,' of ',sim//bach_size)   
    #.print('it should be 981',data.shape)       
    for i in np.arange(1,bach_size):
        new  = gibbs(current,train0,j,v,k,y01)
        '''Updating chain'''
        print('------------', i, ' of ',bach_size) 
        if i%10==0:
            count_s1+=1
            chain_la_sk[:,count_s1]=new.la_sk.reshape(1,-1)
            chain_la_cj[count_s1]=new.la_cj.tolist()
            chain_la_ev[count_s1]=new.la_ev.tolist()
            if i%20==0:
                count_s2+=1
                chain_lm_tht[:,count_s2]=new.lm_tht.reshape(1,-1)
                chain_lm_phi[:,count_s2]=new.lm_phi.reshape(1,-1)
                #if i%100 == 0: 
                    #print('iteration ',ite, 'bach ', i) 
        current = new 

    np.savetxt('Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_cj, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_laev_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_ev, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')
#    accuracy(ite,id,data,data.shape[0],k,y01)


print("--- %s min ---" % int((time.time() - start_time)/60))
print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))


