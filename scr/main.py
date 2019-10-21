'''Loading libraries'''
import pandas as pd 
import numpy as np 
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score
import copy
import sys 
import os
from sklearn.metrics.pairwise import cosine_similarity
path = 'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019'
sys.path.append(path+'\\scr')
import functions as fc
import plots as pl
os.chdir(path)


'''
Note: 
    - Fitting the outcome model 
    - first attempt will be a combine model with original features and latent features, in a rf, nb, nn and lr
    - model id = '12'
    - test plots 
    - testing testing set predictions

'''
'''Parameters'''
class parameters:
    __slots__ = ( 'la_cj','la_sk','la_ev','lm_phi','lm_tht')   
    def __init__(self,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format 


'''Hyperparameters'''
k = 30 #Latents Dimension 
sim = 1000 #Simulations 
bach_size = 200 #Batch size for memory purposes 
step1 = 10 #Saving chain every step1 steps 
id = '13' #identification of simulation 
simulations = 1

'''Loading dataset'''
filename = "data\\tcga_train_gexpression.txt"
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_ge_balanced.txt"
   
data = pd.read_csv(filename, sep=';')
data = data.iloc[:, 0:300]


f1_sample = []
acc_sample = []    
for experiment in np.arange(0,simulations):  
    print('Experiment ', experiment, ' of 100')
    '''Splitting Dataset'''
    train, test = train_test_split(data, test_size=0.3) #random_state=22
    #train = data
    
    '''Organizing columns names'''
    remove = train.columns[[0,1]]
    y = train.columns[1]
    y01 = np.array(train[y])
    train = train.drop(remove, axis = 1)
    y01_t = np.array(test[y])
    test = test.drop(remove, axis = 1)
    train0 = np.matrix(train)
    
    
    '''Defining variables'''
    v = train.shape[1] #genes
    j = train.shape[0] #patients 
    
    
    '''Initial Values'''
    current = parameters(np.repeat(0.5,j), #la_cj 0.25
                       np.repeat(150.5,k*2).reshape(2,k), #la_sk 62
                       np.repeat(1.0004,v), #la_ev FIXED
                       np.repeat(1/v,v*k).reshape(v,k),#lm_phi v x k 
                       np.repeat(150.5,j*k).reshape(j,k)) #lm_theta k x j
                       #np.repeat(0.5, j), #la_pj
                       #np.repeat(0.5,k)) #la_qk 
     
    
    '''Creating the chains'''
    chain_la_sk = np.tile(current.la_sk.reshape(-1,1),(1,int(bach_size/step1)))
    chain_la_cj = np.tile(current.la_cj.reshape(-1,1),(1,int(bach_size/step1)))
    chain_lm_tht = np.tile(current.lm_tht.reshape(-1,1),(1,int(bach_size/step1)))
    chain_lm_phi = np.tile(current.lm_phi.reshape(-1,1),(1,int(bach_size/step1)))
    
    '''Sampling'''
    start_time = time.time()    
        
    for ite in np.arange(0,sim//bach_size):    
        count_s1 = 0
        count_s2 = 0
        chain_la_sk[:,count_s1]=current.la_sk.reshape(1,-1)
        chain_la_cj[:,count_s1]=current.la_cj.reshape(1,-1)
        chain_lm_tht[:,count_s1]=current.lm_tht.reshape(1,-1)
        chain_lm_phi[:,count_s1]=current.lm_phi.reshape(1,-1)
        print('iteration--',ite,' of ',sim//bach_size)   
        #.print('it should be 981',data.shape)       
        for i in np.arange(1,bach_size):
            new  = fc.gibbs(current,train0,j,v,k,y01)
            '''Updating chain'''
            if i%10==0:
                #print('------------', i, ' of ',bach_size) 
                count_s1+=1
                chain_la_sk[:,count_s1]=new.la_sk.reshape(1,-1)
                chain_la_cj[:,count_s1]=new.la_cj.reshape(1,-1)
                chain_lm_tht[:,count_s1]=new.lm_tht.reshape(1,-1)
                chain_lm_phi[:,count_s1]=new.lm_phi.reshape(1,-1)
                if i%90 == 0: 
                    test1 = np.dot(current.lm_tht,np.transpose(current.lm_phi))
                    print(test1.mean(), train0.mean())

    
            current= copy.deepcopy(new )  
        np.savetxt('results\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
        np.savetxt('results\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_cj, delimiter=',',fmt='%5s')
        np.savetxt('results\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
        np.savetxt('results\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')
    
    
    print("--- %s min ---" % int((time.time() - start_time)/60))
    print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))
    
    
    '''Loading average values back for predictions'''
    ite0 = 2
    la_sk = np.loadtxt('results\\output_lask_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    la_cj = np.loadtxt('results\\output_lacj_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    lm_phi = np.loadtxt('results\\output_lmphi_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    lm_tht = np.loadtxt('results\\output_lmtht_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    
    
    for ite in np.arange(ite0+1,sim//bach_size):
        la_sk = la_sk + np.loadtxt('results\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        la_cj = la_cj + np.loadtxt('results\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        lm_phi = lm_phi + np.loadtxt('results\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        lm_tht = lm_tht + np.loadtxt('results\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
    
    la_sk = la_sk/((sim//bach_size)-1)
    la_cj = la_cj/((sim//bach_size)-1)
    lm_phi = lm_phi/((sim//bach_size)-1)
    lm_tht = lm_tht/((sim//bach_size)-1)
    
    
    la_sk = la_sk.reshape(2,k)
    la_cj = la_cj.reshape(j,1)
    lm_tht = lm_tht.reshape(j,k)
    lm_phi = lm_phi.reshape(v,k)
    
    #print(la_cj.shape, la_sk.shape,la_sk[0,:].shape ,lm_tht.shape)
    #print(current.la_cj.shape, current.la_sk.shape,current.la_sk[0,:].shape ,current.lm_tht.shape)
    
    #print('final accucary on the average values sampled')
    #acc(lm_tht,la_sk,la_cj,y01)
    #acc(current.lm_tht,current.la_sk,current.la_cj,y01)
    
    '''Predictions on testing set
    lm_tht_pred = np.repeat(0.5,test.shape[0]*k).reshape(test.shape[0],k)
    test0 = np.matrix(test) 
    
    for j in np.arange(test.shape[0]):
        # intialise data of lists. 
        sim_list = list(cosine_similarity(test0[j,:], train0)[0])
        sim_list= pd.DataFrame({'sim':sim_list})
        sim_list = sim_list.sort_values(by=['sim'],  ascending=False)
        lm_tht_pred[j,:] = lm_tht[list(sim_list.index[0:6])].mean(axis=0)         
    
    y01_t_p = fc.PGM_pred(lm_tht_pred,la_sk,la_cj,y01_t)
    ac = confusion_matrix(y01_t, y01_t_p)
    acc_sample.append((ac[0,0]+ac[1,1])/ac.sum())    
    f1_sample.append(f1_score(y01_t, y01_t_p))
        
    with open('pgm_id12_f1.txt', 'w') as f:
        for item in f1_sample:
            f.write("%s\n" % item)
    
    with open('pgm_id12_acc.txt', 'w') as f:
        for item in acc_sample:
            f.write("%s\n" % item)'''
    fc.predictions_test(test,train0,y01_t,lm_tht,la_sk,la_cj,k)


print('acc: ',acc_sample)
print('f1 : ', f1_sample)




'''
#PLOTS 

pl.plot_chain_sk('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lask_id',sim//bach_size, 15,id)
pl.plot_chain_cj('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lacj_id',sim//bach_size, 15)
pl.plot_chain_tht('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmtht_id',sim//bach_size, 15)
pl.plot_chain_phi('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmphi_id',sim//bach_size, 15)

'''
