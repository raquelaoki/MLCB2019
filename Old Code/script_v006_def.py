import numpy as np 
import math
import pandas as pd 
from scipy.stats import dirichlet, beta, nbinom, norm, gamma
#from scipy.special import loggamma,gamma
import gc
import json
import random 
import matplotlib.pyplot as plt
from sklearn import metrics


#IMPLEMENTING GIBBS SAMPLER 

'''Parameters'''
class parameters:
    __slots__ = ('ln', 'la_cj','la_sk','la_ev','lm_phi','lm_tht')   
    def __init__(self, latent_v,latent_cj,latent_sk,latent_ev,latent_phi ,latent_tht):
        self.ln = latent_v #array with parameters that are only one number [0-c0,1-gamma0]
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #string of array K
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format      
        #self.p = prediction #string of array [intercept, gender, 15 cancer types, k genes]



'''Ratio functions'''
#np.exp(max)
def gibs(p_cur, data_F,k,y):
    '''Priori Ration'''
    #J is samples and V is genes
    j = data_F.shape[0]
    v = data_F.shape[1]-1
    y01 = data_F['y']
    data_F = data_F.drop(y,axis = 1)
    p_new = p_cur.copy()
    p_new.la_ev = np.random.gamma(np.sqrt(p_cur.la_ev),np.sqrt(p_cur.la_ev),1)
    p_new.lm_phi = np.random.dirichlet(p_new.la_ev,k)
    p_new.la_cj = np.random.gamma(np.sqrt(np.sqrt(v*7.42)))


#https://appsilon.com/how-to-sample-from-multidimensional-distributions-using-gibbs-sampling/
    
    return p_new



'''
Creatint the MCMC for the model
MCMC(
startvalue = initial value for the parameters
iterations = 
data = complete data with all columns 
k = number of latent variables
remove, lr, y = columns names to be removed, presente only in the logistic regression part and y
)
'''
       

'''MCMC algorithm'''
def MCMC(startvalue, #start value of the chain 
         bach_size, #bach size for save files 
         data, #full dataset
         k, #size of latent features
         lr, #column names for the logistc regression 
         y01,  #metastase 0/1 array
         id, #id of the attempt 
         ite, #ite in sim/bach 
         step1, step2, #frequency i save values on array
         c_ln,c_la_sk,c_la_cj, c_la_ev, #array with the chain of values step1
         c_lm_tht,c_lm_phi): #array with the chain of values step2
    '''Splitting dataset'''
    data_F = data.drop(lr,axis = 1)

    '''Tracking acceptance rate and steps count'''
    a_F = 0
    
    count_s1 = 0
    count_s2 = 0
    
    '''Starting chain and parametrs'''
    param_cur = startvalue  
    c_ln[count_s1]=param_cur.ln.tolist()
    c_la_sk[:,count_s1]=param_cur.la_sk.reshape(1,-1)
    c_la_cj[count_s1]=param_cur.la_cj.tolist()
    c_la_ev[count_s1]=param_cur.la_ev.tolist()
    c_lm_tht[:,count_s2]=param_cur.lm_tht.reshape(1,-1)
    c_lm_phi[:,count_s2]=param_cur.lm_phi.reshape(1,-1)
    
    #print('inside the MCMC',c_la_sk.shape)
    
    for i in np.arange(1,bach_size):
        '''Factor Analysis - Latent Features'''
        param_new_f = proposal_f(param_cur)
#        if i%100 == 0: 
#            a = a_F*100/i
#            b = a_P*100/i
#            print('iteration ',ite, 'bach i', i,' acceptance ', "%0.2f" % a,'-', "%0.2f" % b)

        prob_f = np.exp(ration(param_new_f,param_cur, data_F,k,y))
        if np.random.uniform(0,1,1)<prob_f:
            param_cur = param_new_f
            a_F+=1
            
        '''Updating chain'''
        if i%10==0:
            count_s1+=1
            #c_p[count_s1]=param_cur.p.tolist()
            c_ln[count_s1]=param_cur.ln.tolist()
            c_la_sk[:,count_s1]=param_cur.la_sk.reshape(1,-1)
            c_la_cj[count_s1]=param_cur.la_cj.tolist()
            c_la_ev[count_s1]=param_cur.la_ev.tolist()

            if i%20==0:
                count_s2+=1
                c_lm_tht[:,count_s2]=param_cur.lm_tht.reshape(1,-1)
                c_lm_phi[:,count_s2]=param_cur.lm_phi.reshape(1,-1)
                if i%100 ==0: 
                    a = a_F*100/i
                    print('iteration ',ite, 'bach i', i,' acceptance ', "%0.2f" % a)
    
                           
 
    #np.savetxt('Data\\output_p_id'+str(id)+'_bach'+str(ite)+'.txt', c_p, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_ln_id'+str(id)+'_bach'+str(ite)+'.txt', c_ln, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', c_la_sk, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', c_la_cj, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_laev_id'+str(id)+'_bach'+str(ite)+'.txt', c_la_ev, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', c_lm_tht, delimiter=',',fmt='%5s')
    np.savetxt('Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', c_lm_phi, delimiter=',',fmt='%5s')
    accuracy(ite,id,data,data.shape[0],k,y01)
    return param_cur, a_F

'''
function to check the quality of the LR predictions
iteration: refers to iterations between number of total simulations and batchs
id: id of the simulation
the output is a print
'''        
def accuracy(iteration,id,data,j,k,y01):
    files_sk = []
    files_cj = []
    files_tht = []
    #data2 = data.copy()
    files_sk.append('Data\\output_lask_id'+id+'_bach'+str(0)+'.txt')
    files_cj.append('Data\\output_lacj_id'+id+'_bach'+str(0)+'.txt')
    files_tht.append('Data\\output_lmtht_id'+id+'_bach'+str(0)+'.txt')
    sk_sim=pd.read_csv(files_sk[0],sep=',', header=None)      
    cj_sim=pd.read_csv(files_cj[0],sep=',', header=None)      
    tht_sim=pd.read_csv(files_tht[0],sep=',', header=None)      
    
    if iteration >=1 :
        for ite in range(iteration):
            files_sk.append('Data\\output_lask_id'+id+'_bach'+str(ite)+'.txt')
            files_cj.append('Data\\output_lacj_id'+id+'_bach'+str(ite)+'.txt')
            files_tht.append('Data\\output_lmtht_id'+id+'_bach'+str(ite)+'.txt')
        
        #Loading files
        for i in range(1,len(files_sk)):
            sk_sim = pd.concat([sk_sim,pd.read_csv(files_sk[i],sep=',', header=None)],axis=1) 
            cj_sim = pd.concat([cj_sim,pd.read_csv(files_cj[i],sep=',', header=None)],axis=0) 
            tht_sim = pd.concat([tht_sim,pd.read_csv(files_tht[i],sep=',', header=None)],axis=1) 
        #phi: every column is a simulation, every row is a position in the matrix
        #removing the first 20% as burn-in phase
    tht_array = []
    sk_array = []
    for i in range(20,tht_sim.shape[1]):
        tht_array.append(np.array(tht_sim.iloc[0:,i]).reshape(j,k))
    theta = np.mean( tht_array , axis=0 )
    sk = np.matrix(np.mean( sk_sim , axis=1 )).reshape(k,sk_sim.shape[0]//k)
    cj = np.matrix(np.mean(cj_sim, axis=0))    
        #data_NB = pd.DataFrame(theta)
    fit = []
    
    for i in range(theta.shape[0]):
        pred0 = np.log(norm.pdf(x = theta[i,],loc = sk[:,0],scale = cj[:,i])).sum()
        pred1 = np.log(norm.pdf(x = theta[i,],loc = sk[:,1],scale = cj[:,i])).sum()
        print(pred0,pred1)
        if pred0>=pred1:
            fit.append(0)
        else:
            fit.append(1)

    #fit = 1/(1+np.exp(data_P.mul(p).sum(axis=1))) 
    tn, fp, fn, tp = metrics.confusion_matrix(data['y'], fit, labels=None, sample_weight=None)
    print('Tn',tn,'Tp',tp,'Fn',fn,'Fp',fp,)
    
    


'''
print some plots to check the convergence of the parameters
Options parameter: lask, lacj, laev, ln, p
Options not implemented yet: lmphi and lmtht
'''    
def conv_plots(sim,bach_size,parameter,id):
    files = []
    for ite in range(sim//bach_size):
        files.append('Data\\output_'+parameter+'_id'+id+'_bach'+str(ite)+'.txt')
    
    f=pd.read_csv(files[0],sep=',', header=None)
    for i in range(1,len(files)):
        f = pd.concat([f,pd.read_csv(files[i],sep=',', header=None)],axis =0,sort=False)
    
    
    '''Plots'''
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
