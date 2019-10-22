'''Loading libraries'''
import pandas as pd 
import numpy as np 
import sys 
import os
path = 'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019'
sys.path.append(path+'\\scr')
import functions as fc
os.chdir(path)


'''
Flags
'''
RUN_MCMC = False
RUN_OUTCOME = False
RUN_PREDICTIONS = False

if RUN_MCMC:  #simulations to have an ic for acc and f1
    simulations = 1#00 
else: 
    simulations = 1


'''
Note: 
    - Fitting the outcome model 
    - first attempt will be a combine model with original features and latent features, in a rf, nb, nn and lr
    - model id = '12'
    - test outcome 
    - explain how mcmc save values
    - add the help explaining how to use the function 
'''

'''Hyperparameters'''
k = 30 #Latents Dimension 
sim = 1000 #Simulations 
bach_size = 200 #Batch size for memory purposes 
step1 = 10 #Saving chain every step1 steps 
id = '13' #identification of simulation 

'''Loading dataset'''
filename = "data\\tcga_train_gexpression.txt"

'''Parameters'''
class parameters:
    __slots__ = ( 'la_cj','la_sk','la_ev','lm_phi','lm_tht')   
    def __init__(self,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format 


   
data = pd.read_csv(filename, sep=';')
data = data.iloc[0:500, 0:100]
   
for experiment in np.arange(0,simulations):  
    print('Experiment ', experiment, ' of 100')
    train0, test, j, v, y01, y01_t = fc.mcmc(data,sim,bach_size,step1,k,id,RUN_MCMC)
       
    '''Loading average values back for predictions'''
    la_sk,la_cj,lm_tht,lm_phi = fc.load_chain(id,sim,bach_size,j,v,k)
    fc.predictions_test(test,train0,y01_t,lm_tht,la_sk,la_cj,k,RUN_PREDICTIONS)


'''
PLOTS: evaluating the convergency  
'''
#pl.plot_chain_sk('results\\output_lask_id',sim//bach_size, 15,id)
#pl.plot_chain_cj('results\\output_lacj_id',sim//bach_size, 15)
#pl.plot_chain_tht('results\\output_lmtht_id',sim//bach_size, 15)
#pl.plot_chain_phi('results\\output_lmphi_id',sim//bach_size, 15)



'''Outcome Model'''
c,f = fc.OutcomeModel('rf',train0,lm_tht,y01)