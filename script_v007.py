'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import copy

'''
Notes:
- DATALIMIT: V = 3500 WITH EVERYTHING ELSE CLOSED 
- theta is going crazy 
- compute canada: problem with files location 

'''




'''Hyperparameters'''
k = 100 #Latents Dimension 
sim = 600 #Simulations 
bach_size = 200 #Batch size for memory purposes 
step1 = 10 #Saving chain every step1 steps 
step2 = 20
id = '02' #identification of simulation 

#WRONG< UPDATE HERE 
if bach_size//step2 <= 20:
    print('ERROR ON MCMC, this division must be bigger than 20')

'''Loading dataset'''
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_filtered.txt"
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_binary.txt"
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_gexpression.txt"
data = pd.read_csv(filename, sep=';')


'''Splitting Dataset'''
train, test = train_test_split(data, test_size=0.3, random_state=22)

'''Organizing columns names'''
remove = train.columns[[0,1]]
y = train.columns[1]
y01 = np.array(train[y])
train = train.drop(remove, axis = 1)
y01_t = np.array(test[y])
test = test.drop(remove, axis = 1)


'''Defining variables'''
v = train.shape[1] #genes
j = train.shape[0] #patients 


'''Parameters'''
class parameters:
    __slots__ = ('ln', 'la_cj','la_sk','la_c1','la_ev','lm_phi','lm_tht', 'la_pj', 'la_qk')   
    def __init__(self, latent_v,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht, latent_pj, latent_qk):
        self.ln = latent_v #array with parameters that are only one number [0-c0,1-gamma0]
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format 
        self.la_pj = latent_pj
        self.la_qk = latent_qk

#need to update these values considering the new dataset 
current = parameters(np.repeat(1.65,2),#ln [0-c0,1-gamma0]
                   np.repeat(13.5,j), #la_cj FIXED 
                   np.repeat(14.2,k*2).reshape(2,k), #la_sk 2.23
                   np.repeat(1.0004,v), #la_ev FIXED
                   np.repeat(1/v,v*k).reshape(v,k),#lm_phi v x k 
                   np.repeat(223,j*k).reshape(j,k), #lm_theta k x j
                   np.repeat(0.5, j), #la_pj
                   np.repeat(0.1,k)) #la_qk 

'''Gibbs Sampling'''
train0 = np.matrix(train)

def gibbs(current,train0,j,v,k,y01):
    new = copy.deepcopy(current) 
    #lvk = np.zeros((v,k))
    #ljk = np.zeros((j,k))
    lvjk = np.zeros((v,j,k))
    
    for ki in np.arange(k):
        #for ji in np.arange(j):
            #for vi in np.arange(v):
               # ljvk[ji,vi,ki] = np.random.poisson(current.la_pj[ji]*current.lm_phi[vi,ki]*current.lm_tht[ji,ki])
        lvjk[:,:,ki] = np.dot(current.lm_phi[:,ki].reshape(v,1),
            np.transpose(np.multiply(current.lm_tht[:,ki].reshape(j,1),current.la_pj.reshape(j,1)).reshape(j,1)))       

        
        #ldotdotk = np.multiply(np.array(current.lm_tht[:,ki].reshape(j,1)),current.lm_phi[:,ki].reshape(1,v))
        #ldotdotk = np.multiply(ldotdotk,train0).sum()
        #lvk[:,ki] = np.random.multinomial(ldotdotk,current.lm_phi[:,ki])
        #ljk[:,ki] = np.random.poisson(current.lm_tht[:,ki])
    
    #check sum of poisson. I might be able to apply poisson after the sum, so will be faster
    lvjk = np.random.poisson(lvjk)
    lvk = lvjk.sum(axis=1)
    ljk = lvjk.sum(axis=0)
    ldotdotk = ljk.sum(axis=0)
    for ki in np.arange(k):    
        new.lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+current.la_ev),size = 1)
        new.lm_tht[:,ki] = np.random.gamma(shape=(current.la_sk[y01,ki]+ljk[:,ki]).reshape(j),
                  scale=(current.la_cj.reshape(j)-np.log(1-current.la_pj).reshape(j)))
    
    new.la_qk = np.random.beta(a = ldotdotk, b = v*current.la_ev.mean())
    
    new.la_pj = np.random.beta(a= (1+train0.sum(axis = 1)).reshape(j,1) ,b=(1+new.lm_tht.sum(axis =1)).reshape(j,1))
    #a1 = 2
    #b1 = 2.23    
    #new.la_cj = 1/np.random.gamma(shape = (current.la_sk.sum(axis = 1)[y01]+a1), scale = 1/(b1+new.lm_tht.sum(axis=1)))    
    #g1 = 1
    #c1 = 1
    a2 = 1.5
    b2 = 1.5
    #uvk  = np.repeat(0,v)    
    for ki in np.arange(k):       
        #for vi in np.arange(v):
            #p = current.la_ev[vi]/(current.la_ev[vi]+np.arange(max(lvk[vi,ki],1))+1)
            #uvk[vi] = uvk[vi]+ np.random.binomial(n=1,p=p).sum()             
        uk = np.array([0,0])
        for ji in np.arange(j):
            p = current.la_sk[y01[ji],ki]/(current.la_sk[y01[ji],ki]+np.arange(max(ljk[ji,ki],1))+1)
            uk[y01[ji]] = uk[y01[ji]] +np.random.binomial(1,p=p).sum()
        new.la_sk[:,ki] = 1/np.random.gamma(a2+uk,1/(b2+new.la_qk[ki]))     
    
    #new.la_ev = np.random.gamma(shape = (g1+uvk),scale = c1-v*((np.log(1-new.la_qk)).sum()))
    return(new)

start_time = time.time()    


new  = gibbs(current,train0,j,v,k,y01)
print('tht',current.lm_tht.mean(),new.lm_tht.mean())
print('phi',current.lm_phi[0:10,1],new.lm_phi[0:10,1])
print('sk',current.la_sk.mean(),new.la_sk.mean())
print('cj',current.la_cj.mean(),new.la_cj.mean())
print('ev',current.la_ev.mean(),new.la_ev.mean())
print('pj',current.la_pj.mean(),new.la_pj.mean())
print('qk',current.la_qk.mean(),new.la_qk.mean())
#current= copy.deepcopy(new )


def acc(theta,sk,y):
    y0 = gamma.pdf(theta,current.la_sk[0,:],2.33)
    y1 = gamma.pdf(theta,current.la_sk[1,:],2.33)
    y2 = np.divide(y0,y1)
    y2 = np.nan_to_num(y2, 0.0)
    y2 = y2.prod(axis=1)
    y2[y2<=1] = 0
    y2[y2<=1] = 0
    print(confusion_matrix(y,y2))
    

'''Creating the chains'''
chain_la_sk = np.tile(current.la_sk.reshape(-1,1),(1,int(bach_size/step1)))
chain_la_pj = np.tile(current.la_pj.tolist(),(int(bach_size/step1),1))
chain_la_qk = np.tile(current.la_qk.tolist(),(int(bach_size/step1),1))
chain_lm_tht = np.tile(current.lm_tht.reshape(-1,1),(1,int(bach_size/step2)))
chain_lm_phi = np.tile(current.lm_phi.reshape(-1,1),(1,int(bach_size/step2)))

'''Starting chain and parametrs'''

start_time = time.time()    
    
for ite in np.arange(0,sim//bach_size):    
    count_s1 = 0
    count_s2 = 0
    chain_la_sk[:,count_s1]=current.la_sk.reshape(1,-1)
    chain_la_pj[count_s1]=current.la_pj.reshape(j)
    chain_la_qk[count_s1]=current.la_qk.reshape(k)
    chain_lm_tht[:,count_s2]=current.lm_tht.reshape(1,-1)
    chain_lm_phi[:,count_s2]=current.lm_phi.reshape(1,-1)
    print('iteration--',ite,' of ',sim//bach_size)   
    #.print('it should be 981',data.shape)       
    for i in np.arange(1,bach_size):
        new  = gibbs(current,train0,j,v,k,y01)
        '''Updating chain'''
        if i%10==0:
            print('------------', i, ' of ',bach_size) 
            count_s1+=1
            chain_la_sk[:,count_s1]=new.la_sk.reshape(1,-1)
            chain_la_pj[count_s1]=new.la_pj.reshape(j)
            chain_la_qk[count_s1]=new.la_qk.reshape(k)
            if i%20==0:
                count_s2+=1
                chain_lm_tht[:,count_s2]=new.lm_tht.reshape(1,-1)
                chain_lm_phi[:,count_s2]=new.lm_phi.reshape(1,-1)
                if i%100 == 0: 
                    #print('iteration ',ite, 'bach ', i) 
                    print(acc(current.lm_tht,current.la_sk,y01))
        current= copy.deepcopy(new )

    np.savetxt('Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_pj, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_laev_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_qk, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')
#    accuracy(ite,id,data,data.shape[0],k,y01)


print("--- %s min ---" % int((time.time() - start_time)/60))
print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))

#4:13pm
