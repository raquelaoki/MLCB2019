import numpy as np 
import pandas as pd 
#from scipy.stats import dirichlet, beta, nbinom, norm
#from scipy.special import loggamma,gamma
#import gc
#import json
#import random 
#import matplotlib.pyplot as plt
#from sklearn import metrics
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix,f1_score
#from scipy.stats import gamma
import copy
#from sklearn.metrics.pairwise import cosine_similarity
import os 

'''Parameters'''
class parameters:
    __slots__ = ( 'la_cj','la_sk','la_ev','lm_phi','lm_tht')   
    def __init__(self,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.la_cj = latent_cj #string of array J 
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format 


'''
Gibbs Sampling
Parameters: 
    current: class parameters
    train0: traning set 
    j patients, v genes, k latent features
    y01: array true label
Return: 
    new: class parameters
'''

def gibbs(current,train0,j,v,k,y01):
    new = copy.deepcopy(current) 
    lvjk = np.zeros((v,j,k))
    
    for ki in np.arange(k):
        #0.79 decrease, 0.8 increase   #0.795 it's perferct for k=100
        lvjk[:,:,ki] = np.dot(0.795*current.lm_phi[:,ki].reshape(v,1), current.lm_tht[:,ki].reshape(1,j))       
    #check sum of poisson. I might be able to apply poisson after the sum, so will be faster
    #lvjk = np.random.poisson(lvjk)
    lvk = np.random.poisson(lvjk.sum(axis=1))
    ljk = np.random.poisson(lvjk.sum(axis=0))
    for ki in np.arange(k):    
        new.lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+current.la_ev),size = 1)
        #new.lm_tht[:,ki] = np.random.gamma(shape=(current.la_sk[y01,ki]+ljk[:,ki]).reshape(j),
        #          scale=(np.divide(current.la_cj,1-current.la_cj)).reshape(j))
        new.lm_tht[:,ki] = np.random.gamma(shape=(current.la_sk[y01,ki]+ljk[:,ki]).reshape(j), 
                  scale=np.repeat(0.5,j).reshape(j))
    
    #one for y=0 and another for y=1    
    lk1 = np.dot(y01,ljk)
    lk0 = np.dot(1-y01,ljk)
    a2 = 187
    b2 = 0.8
    c2 = y01.sum()
    new.la_sk[0,:] = np.random.gamma((a2/k)+lk0/(j-c2),b2+0.69)     
    new.la_sk[1,:] = np.random.gamma((a2/k)+lk1/c2,b2+0.69)     
        
    #a1 = 40#4000
    #b1 = 100#10000
    #c1 = 1/1000
    #new.la_cj = np.random.beta(a= (a1+c1*train0.sum(axis = 1)).reshape(j,1) ,b=(b1+c1*new.lm_tht.sum(axis =1)).reshape(j,1))
    new.la_cj = np.repeat(0.5,j).reshape(j,1)
    return(new)

'''
Accuracy
Parameters: 
    theta: np matrix 
    sk: np matrix 
    cj: array (constant)
    y: true label
Return: 
    confusion matrix 
'''
def acc(theta,sk1,cj,y):
    y0 = gamma.logpdf(x=theta,a = sk1[0,:],scale = 1)
    y1 = gamma.logpdf(x=theta,a = sk1[1,:],scale = 1)
    y3 = y1-y0
    y3 = y3.sum(axis=1)
    y3[y3<=0] = 0
    y3[y3>0] = 1
    #print(f1_score(y,y3))
    return confusion_matrix(y,y3)

'''
Predictions
Parameters: 
    theta: np matrix 
    sk: np matrix 
    cj: array (constant)
    y: true label
Return: 
    Predictions
 '''
def PGM_pred(theta,sk1,cj,y):
    y0 = gamma.logpdf(x=theta,a = sk1[0,:],scale = 1)
    y1 = gamma.logpdf(x=theta,a = sk1[1,:],scale = 1)
    y3 = y1-y0
    y3 = y3.sum(axis=1)
    y3[y3<=0] = 0
    y3[y3>0] = 1
    #print(f1_score(y,y3))
    return y3



'''
Pre-Processing driver genes Intogen 
#This first dataset only has information about: 
#BLCA, BRCA ,ESCA,HNSC , LGG, LUSC,PAAD,PRAD 
#missing: ACC, CHOL, LIHC, SARC, SKCM, TCGT, UCS


'''
def preprocessing_dg1(name):
    remove = ["known_match","chr","strand_orig",'info', 'region', 'strand', 'transcript', 'cdna','pos','ref', 
          'alt', 'default_id', 'input','transcript_aminoacids', 'transcript_exons','protein_change',
          'exac_af', 'Pfam_domain', 'cadd_phred',"gdna","protein","consequence","protein_pos",
          "is_in_delicate_domain","inframe_driver_mut_prediction","is_in_cluster","missense_driver_mut_prediction",
          "driver_mut_prediction","mutation_location","disrupting_driver_mut_prediction",'exon',
          "alt_type","known_oncogenic","known_predisposing",'sample','driver_statement']
    files = []
    path = 'data\\driver_genes\\'

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.tsv' in file:
                files.append(os.path.join(r, file))
    
    count = 0
    for f in files: 
        dgenes0 = pd.read_csv(f, sep='\t')
        dgenes0 = dgenes0.drop(remove, axis = 1)
        if count==0:
            dgenes = dgenes0
            count = 1
        else: 
            dgenes = dgenes.append(pd.DataFrame(data = dgenes0), ignore_index=True)


    dgenes = dgenes.sort_values(by=['gene'])
    dgenes = dgenes.dropna(subset=['driver_gene','driver','gene_role'])
    dgenes.drop_duplicates(keep = 'first', inplace = True)  
    cancer_list = ["ACC", "BLCA", "BRCA", "CHOL", "ESCA","HNSC", "LGG", "LIHC", "LUSC", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS"]
    dgenes = dgenes.loc[dgenes['cancer'].isin(cancer_list)]
    
    dgenes.to_csv('data\\'+name,index=False)

