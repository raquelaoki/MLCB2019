#
import numpy as np 
import math
import pandas as pd 
from scipy.stats import dirichlet, beta, nbinom, norm
from scipy.special import loggamma,gamma
import gc
import json
import random 
import matplotlib.pyplot as plt
from sklearn import metrics


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





'''
Proposal distribution
'''
def proposal_f(current):
    new = parameters(np.random.normal(current.ln,0.5), #2
                     np.random.normal(current.la_cj,0.05),#j
                     np.random.normal(current.la_sk,0.25),#kx2
                     np.random.normal(current.la_ev,0.00001),#v
                     np.random.normal(current.lm_phi,0.0000005), #remmeber that lm_phi sum up 1 in the line (genes)
                     np.random.normal(current.lm_tht,0.25)) #remember the average value is 7.42)
    #phi and tht can't be negative 
    new.lm_phi[new.lm_phi<0] = 0.0000001 #this number needs to be smaller 
    col_sums = new.lm_phi.sum(axis=0)
    new.lm_phi = new.lm_phi / col_sums[np.newaxis,:]
    new.la_ev[new.la_ev<0.0000001] =0.0000001  
    new.lm_tht[new.lm_tht<0]=0
    new.lm_tht = new.lm_tht+0.000001
    new.la_sk[new.la_sk<0.0000001] = 0.0000001    
    return new


'''Ratio functions'''
#np.exp(max)
def ration(p_new,p_cur, data_F,k,y):
    '''Priori Ration'''
    #J is samples and V is genes
    j = data_F.shape[0]
    v = data_F.shape[1]-1
    y01 = data_F['y']
    data_F = data_F.drop(y,axis = 1)
    A0 = k*(loggamma(np.prod(p_cur.la_ev))-loggamma(np.prod(p_new.la_ev)))
    
    A1 = k*(np.sum(np.log(gamma(p_new.la_ev)))-np.sum(np.log(gamma(p_cur.la_ev))))
    A2 = np.matmul((p_new.la_ev-1),np.log(p_new.lm_phi)).sum()-np.matmul((p_cur.la_ev-1),np.log(p_cur.lm_phi)).sum()
    #B: eta_j~Gamma(a0,b0)
    a0 = 1#/(2*v)
    b0 = 1#/(2*v)
    B = (a0-1)*(np.log(p_new.la_ev)-np.log(p_cur.la_ev)).sum()+(p_cur.la_ev-p_new.la_ev).sum()/b0
    
    '''
    OLD DISTRIBUTION. IT WORKS 
    #C: theta_kl~Gamma(sk,cj)
    C00 = (j-y01.sum())*(loggamma(p_cur.la_sk[0]).sum()-loggamma(p_new.la_sk[0]).sum())+ (
            y01.sum())*(loggamma(p_cur.la_sk[1]).sum()-loggamma(p_new.la_sk[1]).sum())+ (
                    p_cur.la_sk[0].sum()*np.dot(np.log(p_cur.la_cj),1-y01).sum()+p_cur.la_sk[1].sum()*np.dot(
                            np.log(p_cur.la_cj),y01).sum()-
                    p_new.la_sk[0].sum()*np.dot(np.log(p_new.la_cj),1-y01).sum()-p_new.la_sk[1].sum()*np.dot(
                            np.log(p_new.la_cj),y01).sum())
         
    C01 = j*(loggamma(p_cur.la_sk).sum()-loggamma(p_new.la_sk).sum())+(
    p_cur.la_sk.sum()*np.log(np.dot(p_cur.la_cj,1-y01)).sum()-p_new.la_sk.sum()*np.log(np.dot(p_new.la_cj,y01)).sum())
    C0 = C00+C01
    y01 = y01.as_matrix().reshape(1,len(y01))
    C1 = np.matmul(p_new.la_sk[0]-1,(np.log(p_new.lm_tht)*(1-y01)).sum(axis=1))+ np.matmul(
            p_new.la_sk[1]-1,(np.log(p_new.lm_tht)*y01).sum(axis=1))-np.matmul(
            p_cur.la_sk[0]-1,(np.log(p_cur.lm_tht)*(1-y01)).sum(axis=1))-np.matmul(
            p_cur.la_sk[0]-1,(np.log(p_cur.lm_tht)*y01).sum(axis=1))
    C2 = np.divide(p_cur.lm_tht.sum(axis=0),p_cur.la_cj).sum()-np.divide(p_new.lm_tht.sum(axis=0),p_new.la_cj).sum()
    '''
    #C: theta_kl~Normal(sk,cj)
    y01 = y01.as_matrix().reshape(len(y01),1)
    C0a = p_cur.lm_tht - np.repeat(p_cur.la_sk[0],j).reshape(k,j)#np.add(1,np.multiply(-1,y01))   
    C0b = p_new.lm_tht - np.repeat(p_new.la_sk[0],j).reshape(k,j)
    C0 = (-1)*(np.power(C0a,2)/np.power(p_cur.la_cj,2)-np.power(C0b,2)/np.power(
            p_new.la_cj,2).dot(np.add(1,np.multiply(-1,y01)))).sum()
     
    C1a = p_cur.lm_tht - np.repeat(p_cur.la_sk[1],j).reshape(k,j)#np.add(1,np.multiply(-1,y01))   
    C1b = p_new.lm_tht - np.repeat(p_new.la_sk[1],j).reshape(k,j)
    C1 = (np.power(C1a,2)/np.power(p_cur.la_cj,2)-np.power(C1b,2)/np.power(p_new.la_cj,2).dot(y01)).sum()
    
    C2 = k*((np.log(np.power(p_cur.la_cj,2))-np.log(np.power(p_new.la_cj,2))).sum())/2
    #D: sk~Gamma(gamma0,c0), gamma0 = c0 = (v*averageExpression)^0.5
    #7.42 = log(1680)
    average4 = np.sqrt(v*7.42)
    gamma0 = average4
    c0 = average4
    D = (gamma0-1)*(np.log(p_new.la_sk)-np.log(p_cur.la_sk)).sum()+(p_cur.la_sk-p_new.la_sk).sum()/c0
    
    #E: Cj~Gamma(a1,b1)
    a1 = average4
    b1 = average4
    E = (a1-1)*(np.log(p_new.la_cj)-np.log(p_cur.la_cj)).sum()+(p_cur.la_cj-p_new.la_cj).sum()/b1
    
    #F: gamma0~Gamma(a2,b2)
    average8 = np.sqrt(average4)
    a2 = average8
    b2 = average8
    F = (a2-1)*(np.log(p_new.ln[1])-np.log(p_cur.ln[1]))+(p_cur.ln[1]-p_new.ln[1])/b2
    
    #G: c0~Gamma(a3,b3)
    a3 = average8
    b3 = average8
    G = (a3-1)*(np.log(p_new.ln[0])-np.log(p_cur.ln[0]))+(p_cur.ln[0]-p_new.ln[0])/b3
    '''Likelihood OLD
    #I: n_vj~Poisson(phi_vk theta_kj)
    I0 = np.transpose(np.log(np.matmul(p_new.lm_phi,p_new.lm_tht))-np.log(np.matmul(p_cur.lm_phi,p_cur.lm_tht)))
    I1 = np.multiply(data_F.to_numpy(),I0).sum() #as_matrix()
    I2 = (np.matmul(p_cur.lm_phi,p_cur.lm_tht)-np.matmul(p_new.lm_phi,p_new.lm_tht)).sum()
    '''
    
    '''Likelihood '''
    #I: n_vj~Normal(phi_vk theta_kj, sigma), sigma is constant
    sigma = 4
    #aux = p_cur.lm_phi.dot(p_cur.lm_tht)
    #print('test',data_F.to_numpy().shape,aux.shape)
    I1 = (np.power((np.transpose(data_F.to_numpy())-p_cur.lm_phi.dot(p_cur.lm_tht)),2)-(
            np.power(np.transpose(data_F.to_numpy())-p_new.lm_phi.dot(p_new.lm_tht),2))).sum()/(2*sigma^2)
    I2 = 0 # (np.log(sigma')-np.log(sigma)).sum()*(j/2)  variance is constant    
    #print('ratio - F',"%0.2f" % A0,"%0.2f" % A1,"%0.2f" % A2,"%0.2f" % B,"%0.2f" % C0,
    #      "%0.2f" % C1,"%0.2f" % C2,"%0.2f" % D,"%0.2f" % E, "%0.2f" % F,"%0.2f" % G,
    #     "%0.2f" % I1,"%0.2f" % I2,'end',(A0+A1+A2+B+C0+C1+C2+D+E+F+G+I1+I2))
    #print('tracking some problems', "%0.2f" % F,'(F)',"%0.2f" % D,'(D)',"%0.2f" % C0,'(C0)',"%0.2f" % C1,'(C1)')
    return (A0+A1+A2+B+C0+C1+C2+D+E+F+G+I1+I2)



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
         y,  #metastase 0/1 array
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
    #accuracy(ite,id,data,data.shape[0],k,y01)
    return param_cur, a_F

'''
function to check the quality of the LR predictions
iteration: refers to iterations between number of total simulations and batchs
id: id of the simulation
the output is a print
'''        
def accuracy(iteration,id,data,j,k,y01):
    files_p = []
    files_tht = []
    data2 = data.copy()
    files_tht.append('Data\\output_lmtht_id'+id+'_bach'+str(0)+'.txt')
    tht_sim=pd.read_csv(files_tht[0],sep=',', header=None)      
    if iteration >=1 :
        for ite in range(iteration):
            files_tht.append('Data\\output_lmtht_id'+id+'_bach'+str(ite)+'.txt')
        
        #Loading files
        for i in range(1,len(files_p)):
            tht_sim = pd.concat([tht_sim,pd.read_csv(files_tht[i],sep=',', header=None)],axis=1) 
    #phi: every column is a simulation, every row is a position in the matrix
    #removing the first 20% as burn-in phase
    tht_array = []
    for i in range(20,tht_sim.shape[1]):
        tht_array.append(np.array(tht_sim.iloc[0:,i]).reshape(j,k))
    theta = np.mean( tht_array , axis=0 )
       
    data_NB = pd.DataFrame(theta)

    
    fit = 1/(1+np.exp(data_P.mul(p).sum(axis=1)))
    print('Fit Values: ',fit.min(),'(min) ',fit.mean(),'(mean) ',fit.max(),'(max)')
    fit[fit>0.5] = 1
    fit[fit<=0.5] = 0  
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
