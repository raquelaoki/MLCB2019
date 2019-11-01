'''Loading libraries'''
import pandas as pd 
import numpy as np 
import sys 
import os
path = 'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019'
sys.path.append(path+'\\scr')
import functions as fc
os.chdir(path)
from sklearn.metrics import confusion_matrix, f1_score
pd.set_option('display.max_columns', 500)


'''
Flags
'''
RUN_MCMC = True
RUN_OUTCOME = False
RUN_PREDICTIONS = True

if RUN_MCMC:  #simulations to have an ic for acc and f1
    simulations = 1#00 
else: 
    simulations = 1

'''
Note: 
    - model id = '12'
    - outcome model: problem with the nb, how to get coefs and prefictiosn are really bad
    - explain how mcmc save values
    - add the help explaining how to use the function 
    - outcome model should have train and testing set? check with the baseline paper 
    - updating training set to force the presence of known driver genes
    
'''

'''Hyperparameters'''
k = 18 #Latents Dimension 
sim = 1000 #Simulations 
bach_size = 200 #Batch size for memory purposes 
step1 = 10 #Saving chain every step1 steps 
id = '16' #identification of simulation 

'''Loading dataset'''
filename = "data\\tcga_train_gexpression_cgc.txt" #

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
#data = data.iloc[0:500, 0:100]
   
for experiment in np.arange(0,simulations):  
    print('Experiment ', experiment, ' of 100')
    train0, test, j, v, y01, y01_t, abr, abr_t = fc.mcmc(data,sim,bach_size,step1,k,id,RUN_MCMC)
       
    '''Loading average values back for predictions'''
    la_sk,la_cj,lm_tht,lm_phi = fc.load_chain(id,sim,bach_size,j,v,k)
    y01_t_pred = fc.predictions_test(test,train0,y01_t,lm_tht,la_sk,la_cj,k,RUN_PREDICTIONS)

#
print('\nPGM: Metrics training set: \n'
      ,confusion_matrix(y01,fc.PGM_pred(lm_tht,la_sk,la_cj)),'\n',
      f1_score(y01,fc.PGM_pred(lm_tht,la_sk,la_cj) ))
print('\nPGM: Metrics testing set: \n'
      ,confusion_matrix(y01_t,y01_t_pred) ,'\n',
      f1_score(y01_t,y01_t_pred))

'''
PLOTS: evaluating the convergency  
'''
#pl.plot_chain_sk('results\\output_lask_id',sim//bach_size, 15,id)
#pl.plot_chain_cj('results\\output_lacj_id',sim//bach_size, 15)
#pl.plot_chain_tht('results\\output_lmtht_id',sim//bach_size, 15)
#pl.plot_chain_phi('results\\output_lmphi_id',sim//bach_size, 15)



'''Outcome Model'''
c,f,cm = fc.OutcomeModel('rf',train0,lm_tht,y01)
print('\n Metrics OutcomeModel: ','\n',f,'\n',cm,'\n')

c,f,cm = fc.OutcomeModel('lr',train0,lm_tht,y01)
print('\n Metrics OutcomeModel: ','\n',f,'\n',cm,'\n')


'''Driver Genes'''
#cgc_list = fc.cgc()



'''Exploring Outcome Model's results'''
#c_gene = data.drop(['patients','y'],axis=1).columns
#gene_coef = pd.DataFrame({'genes':c_gene,
#                          'coefs': c[0]})

#gene_coef['coefs_abs'] = abs(gene_coef['coefs'] )    
#gene_coef.sort_values(['coefs_abs'], axis = 0, ascending=False,inplace=True)
#gene_coef.head()


#Selecting top 10% (200 genes)
#gene_coef_sub = gene_coef.iloc[0:200,].reset_index( drop='True')

#merging

#gene_coef_sub = pd.merge(gene_coef_sub,cgc_list, left_on = 'genes',right_on='Gene Symbol')

#print(gene_coef_sub.shape)#












