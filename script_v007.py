'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import gamma
import copy

'''
Notes:
- DATALIMIT: V = 3500 WITH EVERYTHING ELSE CLOSED 
- compute canada: problem with files location 
'''
computecanada = False

'''Hyperparameters'''
k = 100 #Latents Dimension 
sim = 1000 #Simulations 
bach_size = 200 #Batch size for memory purposes 
step1 = 10 #Saving chain every step1 steps 
step2 = 20
id = '05' #identification of simulation 

#WRONG< UPDATE HERE 
if bach_size//step2 <= 20:
    print('ERROR ON MCMC, this division must be bigger than 20')

'''Loading dataset'''
if computecanada: 
    filename = "tcga_train_ge_balanced.txt"
else: 
    filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_ge_balanced.txt"
    

data = pd.read_csv(filename, sep=';')

'''Splitting Dataset'''
#train, test = train_test_split(data, test_size=0.3, random_state=22)
train = data

'''Organizing columns names'''
remove = train.columns[[0,1]]
y = train.columns[1]
y01 = np.array(train[y])
train = train.drop(remove, axis = 1)
#y01_t = np.array(test[y])
#test = test.drop(remove, axis = 1)


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
        #self.la_pj = latent_pj
        #self.la_qk = latent_qk

#need to update these values considering the new dataset 
current = parameters(np.repeat(1.65,2),#ln [0-c0,1-gamma0]
                   np.repeat(0.5,j), #la_cj 0.25
                   np.repeat(199.50,k*2).reshape(2,k), #la_sk 62
                   np.repeat(1.0004,v), #la_ev FIXED
                   np.repeat(1/v,v*k).reshape(v,k),#lm_phi v x k 
                   np.repeat(100,j*k).reshape(j,k)) #lm_theta k x j
                   #np.repeat(0.5, j), #la_pj
                   #np.repeat(0.5,k)) #la_qk 

'''Gibbs Sampling'''
train0 = np.matrix(train)

def gibbs(current,train0,j,v,k,y01):
    new = copy.deepcopy(current) 
    lvjk = np.zeros((v,j,k))
    
    for ki in np.arange(k):
        lvjk[:,:,ki] = np.dot(current.lm_phi[:,ki].reshape(v,1), current.lm_tht[:,ki].reshape(1,j))       
    #check sum of poisson. I might be able to apply poisson after the sum, so will be faster
    lvjk = np.random.poisson(lvjk)
    lvk = lvjk.sum(axis=1)
    ljk = lvjk.sum(axis=0)
    for ki in np.arange(k):    
        new.lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+current.la_ev),size = 1)
        new.lm_tht[:,ki] = np.random.gamma(shape=(current.la_sk[y01,ki]+ljk[:,ki]).reshape(j),
                  scale=(np.divide(current.la_cj,1-current.la_cj)).reshape(j))
    
    a2 = 1000000 #12000 before and average of 33
    b2 = 100000000
    #it shoud be +
    b2u = (np.log(np.divide(current.la_cj ,current.la_cj+np.log(1-0.1)))).sum()
    
    for ki in np.arange(k):       
        uk = np.array([0,0])
        for ji in np.arange(j):
            p = current.la_sk[y01[ji],ki]/(current.la_sk[y01[ji],ki]+np.arange(max(ljk[ji,ki],1))+1)
            uk[y01[ji]] = uk[y01[ji]]+np.random.binomial(1,p=p).sum()
        new.la_sk[:,ki] = 1/np.random.gamma(a2+uk,1/(b2-b2u))     
        
    a1 = 4000
    b1 = 10000
    new.la_cj = np.random.beta(a= (a1+train0.sum(axis = 1)).reshape(j,1) ,b=(b1+new.lm_tht.sum(axis =1)).reshape(j,1))
    return(new)

start_time = time.time()    


#new  = gibbs(current,train0,j,v,k,y01)
#print('tht',current.lm_tht.mean(),new.lm_tht.mean())
#print('phi',current.lm_phi[0:10,1],new.lm_phi[0:10,1])
#print('sk',current.la_sk.mean(),new.la_sk.mean())
#print('cj',current.la_cj.mean(),new.la_cj.mean())
#print('ev',current.la_ev.mean(),new.la_ev.mean())
#print('pj',current.la_pj.mean(),new.la_pj.mean())
#print('qk',current.la_qk.mean(),new.la_qk.mean())
#current= copy.deepcopy(new )


def acc(theta,sk,cj,y):
    y0 = gamma.pdf(x=theta,a = sk[0,:],scale = 1/cj)
    y1 = gamma.pdf(x=theta,a = sk[1,:],scale = 1/cj)
    y3 = np.log(y1)-np.log(y0)
    y3 = y3.sum(axis=1)
    y3[y3<=0] = 0
    y3[y3>0] = 1
    print(confusion_matrix(y,y3))
    

'''Creating the chains'''
chain_la_sk = np.tile(current.la_sk.reshape(-1,1),(1,int(bach_size/step1)))
chain_la_cj = np.tile(current.la_cj.tolist(),(int(bach_size/step1),1))
chain_lm_tht = np.tile(current.lm_tht.reshape(-1,1),(1,int(bach_size/step2)))
chain_lm_phi = np.tile(current.lm_phi.reshape(-1,1),(1,int(bach_size/step2)))

'''Starting chain and parametrs'''

start_time = time.time()    
    
for ite in np.arange(0,sim//bach_size):    
    count_s1 = 0
    count_s2 = 0
    chain_la_sk[:,count_s1]=current.la_sk.reshape(1,-1)
    chain_la_cj[count_s1]=current.la_cj.reshape(j)
    #chain_la_qk[count_s1]=current.la_qk.reshape(k)
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
            chain_la_cj[count_s1]=new.la_cj.reshape(j)
            #chain_la_qk[count_s1]=new.la_qk.reshape(k)
            if i%20==0:
                count_s2+=1
                chain_lm_tht[:,count_s2]=new.lm_tht.reshape(1,-1)
                chain_lm_phi[:,count_s2]=new.lm_phi.reshape(1,-1)
                if i%100 == 0: 
                    #print('iteration ',ite, 'bach ', i) 
                    print(acc(current.lm_tht,current.la_sk,current.la_cj, y01))
        #print('tht',current.lm_tht.mean(),new.lm_tht.mean())
        #print('phi',current.lm_phi[0:10,1],new.lm_phi[0:10,1])
        #print('sk',current.la_sk.mean(),new.la_sk.mean())
        #print('cj',current.la_cj.mean(),new.la_cj.mean())
        #print('\n')

        current= copy.deepcopy(new )
    if computecanada: 
        np.savetxt('$SLURM_TMPDIR/Data/output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
        np.savetxt('$SLURM_TMPDIR/Data/output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_cj, delimiter=',',fmt='%5s')
        np.savetxt('$SLURM_TMPDIR/Data/output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
        np.savetxt('$SLURM_TMPDIRD/ata/output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')
    else:
        np.savetxt('Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
        np.savetxt('Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_cj, delimiter=',',fmt='%5s')
        np.savetxt('Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
        np.savetxt('Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')
    

print("--- %s min ---" % int((time.time() - start_time)/60))
print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))


#about 4h for 600sim
if computecanada: 
    print("done!")
else:
    #checking the accuracy
    ite = 1
    la_sk = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
    la_cj = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=0)
    lm_phi = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
    lm_tht = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
    
    
    for ite in np.arange(2,sim//bach_size):
        la_sk = la_sk + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        la_cj = la_cj + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=0)
        lm_phi = lm_phi + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        lm_tht = lm_tht + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
    
    la_sk = la_sk/((sim//bach_size)-1)
    la_cj = la_cj/((sim//bach_size)-1)
    lm_phi = lm_phi/((sim//bach_size)-1)
    lm_tht = lm_tht/((sim//bach_size)-1)
    
    
    la_sk = la_sk.reshape(2,k)
    la_cj = la_cj.reshape(j,1)
    lm_tht = lm_tht.reshape(j,k)
    lm_phi = lm_phi.reshape(v,k)
    
    print(la_cj.shape, la_sk.shape,la_sk[0,:].shape ,lm_tht.shape)
    print(current.la_cj.shape, current.la_sk.shape,current.la_sk[0,:].shape ,current.lm_tht.shape)
    
    acc(lm_tht,la_sk,la_cj,y01)
    
    
'''
y0 = gamma.pdf(x=lm_tht,a=la_sk[0,:],scale=1/la_cj)
y1 = gamma.pdf(x=lm_tht,a=la_sk[1,:],scale=1/la_cj)
y3 = y1-y0
y3 = y3.sum(axis=1)
y3[y3<=0] = 0
y3[y3>0] = 1
print(confusion_matrix(y,y3))

test1 = np.dot(current.lm_tht,np.transpose(current.lm_phi))

test = np.dot(lm_tht,np.transpose(lm_phi))
test[0:5,0:5]
'''