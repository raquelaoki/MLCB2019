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
RUN_PREDICTIONS = False
RUN_LOAD_MCMC = False
RUN_MF = False
'''
Note:
    - model id = '12' on MLCB
    - outcome model: problem with the nb, how to get coefs and prefictiosn are really bad
    - explain how mcmc save values
    - k18 and 18, the latent variables on the NB classify everyone on 1, k17,16 and 17,16, perfect prediction on training set
    - k15 all 1
    - new tests with id 21, my model is bad. I need to fix to make sense, applying the outcome model to a well known model
test
#bart: it's important to standartize my variables 0-1


'''

'''MCMC Hyperparameters'''
k_mf = 20 #Latents Dimension
k_mcmc = 10
sim = 2000 #Simulations
bach_size = 200 #Batch size for memory purposes
step1 = 10 #Saving chain every step1 steps
id = '21' #identification of simulation


#def main():
'''Loading dataset'''
filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2

'''Parameters
class parameters:
    __slots__ = ( 'la_cj','la_sk','la_ev','lm_phi','lm_tht')
    def __init__(self,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.la_cj = latent_cj #string of array J
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format
'''


data = pd.read_csv(filename, sep=';')
#data = data.iloc[0:500, 0:100]
train, train_validation, j, v, y01, abr, hold_mask = fc.holdout(data, 0.001)
fc.mcmc(train,y01,sim,bach_size,step1,k_mcmc,id,RUN_MCMC)
la_sk,la_cj,lm_tht,lm_phi = fc.load_chain(id,sim,bach_size,j,v,k_mcmc,RUN_LOAD_MCMC)
if RUN_LOAD_MCMC:
    Z = lm_tht.dot(np.transpose(lm_phi))
    v_pred, test_result = fc.predictive_check_new(train,Z,True)
    if(test_result):
        print('Predictive Check test: PASS')
        fc.outcome_model(train,lm_tht,y01)
    else:
        print('Predictive Check Test: FAIL')


W, F = fc.matrixfactorization(train,k_mf,RUN_MF)
if RUN_MF:
    #Time consuming
    v_pred, test_result = fc.predictive_check_new(train,W.dot(F),True)
    if(test_result):
        print('Predictive Check test: PASS')
    else:
        print('Predictive Check Test: FAIL')

#Z = W.dot(F)
#X = train
#i = 0
#Z_train, Z_test, X_train, X_test = train_test_split(Z, , test_size=0.2)


    #
    #print('\nPGM: Metrics training set: \n'
    #      ,confusion_matrix(y01,fc.PGM_pred(lm_tht,la_sk,la_cj)),'\n',
    #      f1_score(y01,fc.PGM_pred(lm_tht,la_sk,la_cj) ))
    #print('\nPGM: Metrics testing set: \n'
    #      ,confusion_matrix(y01_t,y01_t_pred) ,'\n',
    #      f1_score(y01_t,y01_t_pred))

    #'''
    #PLOTS: evaluating the convergency
    #'''
    #pl.plot_chain_sk('results\\output_lask_id',sim//bach_size, 15,id)
    #pl.plot_chain_cj('results\\output_lacj_id',sim//bach_size, 15)
    #pl.plot_chain_tht('results\\output_lmtht_id',sim//bach_size, 15)
    #pl.plot_chain_phi('results\\output_lmphi_id',sim//bach_size, 15)



    #'''Outcome Model'''
    #c,f,cm = fc.outcome_model('rf',train0,lm_tht,y01)
    #print('\n Metrics outcome_model: ','\n',f,'\n',cm,'\n')

    #c,f,cm = fc.outcome_model('lr',train0,lm_tht,y01)
    #print('\n Metrics outcome_model: ','\n',f,'\n',cm,'\n')


    #'''Driver Genes'''
    #cgc_list = fc.cgc()



    #'''Exploring Outcome Model's results'''
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


    #   return v1,v2


#if __name__ == "__main__":
#    v1,v2 = main()
