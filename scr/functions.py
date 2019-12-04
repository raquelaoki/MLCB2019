import numpy as np
import pandas as pd
import time
import copy
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gamma
from scipy import sparse, stats
import statsmodels.discrete.discrete_model as sm

#from scipy.stats import dirichlet, beta, nbinom, norm
#from scipy.special import gamma
#import gc
#import json
#import random
#import matplotlib.pyplot as plt
#from sklearn import metrics

'''PART 1: DECONFOUNDER ALGORITHM'''

'''Parameters'''
class parameters:
    __slots__ = ( 'la_cj','la_sk','la_ev','lm_phi','lm_tht')
    def __init__(self,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.la_cj = latent_cj #string of array J
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format

#Delete
def data_holdout(data, alpha):
    '''
    Holdout: keep a few elementos from the factor model,
    used to verify the quality of the latent features
    parameters:
        data: full dataset
        alpha: proportion of elements to remove
    return:
        train: training set withouht these elements
        train_validation: validation/true values holdout
    source code: decofounder tutorial
    '''
    data = data.reset_index(drop=True)
    '''Organizing columns names'''
    remove = data.columns[[0,1,2]]
    y = data.columns[1]
    y01 = np.array(data[y])
    abr = np.array(data[data.columns[2]])
    train = data.drop(remove, axis = 1)
    train = np.matrix(train)

    # randomly holdout some entries of data
    j, v = train.shape
    n_holdout = int(alpha * j * v)

    holdout_row = np.random.randint(j, size=n_holdout)
    holdout_col = np.random.randint(v, size=n_holdout)
    #it shold be only ones
    holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), (holdout_row, holdout_col)),shape = train.shape)).toarray()
    holdout_mask[holdout_mask>1]=1
    train_val = np.multiply(holdout_mask, train)
    train = np.multiply(1-holdout_mask, train)
    return train, train_val, j, v, y01,  abr, holdout_mask

def data_prep(data):
    '''
    Holdout: keep a few elementos from the factor model,
    used to verify the quality of the latent features
    parameters:
        data: full dataset
        alpha: proportion of elements to remove
    return:
        train: training set withouht these elements
        train_validation: validation/true values holdout
    source code: decofounder tutorial
    '''
    data = data.reset_index(drop=True)
    '''Organizing columns names'''
    remove = data.columns[[0,1,2]]
    y = data.columns[1]
    y01 = np.array(data[y])
    abr = np.array(data[data.columns[2]])
    train = data.drop(remove, axis = 1)
    colnames = train.columns
    train = np.matrix(train)

    # randomly holdout some entries of data
    j, v = train.shape

    return train, j, v, y01,  abr, colnames

def gibbs(current,train0,j,v,k,y01):
    '''
    Gibbs Sampling: the math, proposal of values
    Parameters:
        current: class (parameters)
        train0: traning set (np.matrix)
        j patients, v genes, k latent features (int)
        y01: array true label (np.array)
    Return:
        new: class (parameters)
    '''
    new = copy.deepcopy(current)
    lvjk = np.zeros((v,j,k))

    for ki in np.arange(k):
        lvjk[:,:,ki] = np.dot(0.801*current.lm_phi[:,ki].reshape(v,1), current.lm_tht[:,ki].reshape(1,j))
        #0.795 previous value, it works
    lvk = np.random.poisson(lvjk.sum(axis=1))
    ljk = np.random.poisson(lvjk.sum(axis=0))
    #print(new.lm_tht.shape, ljk.shape,j,v,k )
    #print('\nphi:',new.lm_phi.shape)
    for ki in np.arange(k):
        new.lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+current.la_ev),size = 1)
        new.lm_tht[:,ki] = np.random.gamma(shape=(current.la_sk[y01,ki]+ljk[:,ki]).reshape(j),
                  scale=np.repeat(0.5,j).reshape(j))

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

def fa_mcmc(train,y01, sim, bach_size, step1,k,id,run):
    '''
    MCMC: call gibbs function and save proposed parameters in a chain
    Parameters
        data: full dataset before splitting (pd)
        sim, bach_size, step1: simulations, bach size and step size (int)
        k: latent variables size (int)
        id: id of the simulation  (str)

    Return
        save the chain on results folder
    '''
    '''Defining variables'''
    j, v  = train.shape #patients x genes
    if run:
        '''Initial Values'''
        current = parameters(np.repeat(0.5,j), #la_cj 0.25
                           np.repeat(150.5,k*2).reshape(2,k), #la_sk
                           np.repeat(1.0004,v), #la_ev FIXED
                           np.repeat(1/v,v*k).reshape(v,k),#lm_phi v x k
                           np.repeat(150.5,j*k).reshape(j,k)) #lm_theta k x j



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
            for i in np.arange(1,bach_size):
                new  = gibbs(current,train,j,v,k,y01)
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
                        print(test1.mean(), train.mean())


                current= copy.deepcopy(new )
            np.savetxt('results\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
            np.savetxt('results\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_cj, delimiter=',',fmt='%5s')
            np.savetxt('results\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
            np.savetxt('results\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')


        print("--- %s min ---" % int((time.time() - start_time)/60))
        print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))
    return

def load_chain(id,sim,bach_size,j,v,k, run):
    '''
    Load chains of values after MCMC
    paramters:
        id: id of the experiment (str)
        sim, bach_size: simulations and bach size (int)
        j,v,k: patients, genes, latent variables size (int)
    return:
        average parameters predicted without the burn-in period
        la_sk, la_cj, lm_tht, lm_phi: (np.matrix)
    '''
    if run:
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
    else:
        la_sk = []
        la_cj = []
        lm_tht = []
        lm_phi = []
    return la_sk,la_cj,lm_tht,lm_phi

#Failed in run = False
def fa_matrixfactorization(train,k,run):
    '''
    Matrix Factorization to extract latent features
    Parameters:
        train: dataset
        k: latent Dimension
        run: True/False
    Return:
        2 matrices
    '''
    if run:
        model = NMF(n_components=k, init='random') #random_state=0
        W = model.fit_transform(train)
        H = model.components_
    else:
        W, H = [], []
    return W, H

def fa_pca(train,k,run):
    '''
    PCA to extrac latent features
    Parameters:
        train: dataset
        k: latent Dimension
        run: True/False
    Return:
        1 matrix
    '''
    if run:
        # Standardizing the features
        X = StandardScaler().fit_transform(train)
        model = PCA(n_components=k)
        principalComponents = model.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents)
    else:
        principalDf = []
    return principalDf

def fa_a(train,k,run):
    from keras.layers import Input, Dense
    from keras.models import Model
    '''
    Autoencoder to extrac latent features
    Parameters:
        train: dataset
        k: latent Dimension
        run: True/False
    Return:
        1 matrix
    References
    #https://www.guru99.com/autoencoder-deep-learning.html
    #https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    if run:
        x_train, x_test = train_test_split(train, test_size = 0.3,random_state = 22)
        print(x_train.shape, x_test.shape, train.shape)
        ii = x_train.shape[1]
        input_img = Input(shape=(ii,))
        encoding_dim = 20
        encoded = Dense(encoding_dim, activation='sigmoid')(input_img) #change relu
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(ii, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)


        autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
        autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

        encoded_imgs = encoder.predict(train)
        return encoded_imgs


def check_save(z,train,colnames,y01,name,k):
    v_pred, test_result = predictive_check_new(train,z,True)
    if(test_result):
        print('Predictive Check test: PASS')
        resul, output = outcome_model( train,colnames, z,y01)
        #np.savetxt('results\\feature_mf_'+str(k_mf)+'_lr'+'_all'+'.txt', resul, delimiter=',',fmt='%5s')
        resul.to_csv('results\\feature_'+name+'_'+str(k)+'_lr'+'_all'+'.txt', sep=';', index = False)
    else:
        print('Predictive Check Test: FAIL')
        print('Results not saved')


def predictive_check_new(X, Z,run ):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    '''
    This function is agnostic to the method.
    Use a Linear Model X_m = f(Z), save the proportion
    of times that the pred(z_test)<X_m(test) for each feature m.
    Compare with the proportion of the null model mean(x_m(train)))<X_m(test)
    Create an Confidence interval for the null model, check if the average value
    across the predicted values using LM is inside this interval

    Sample a few columns (300 hundred?) to do this math

    Parameters:
        X: orginal features
        Z: latent (either the reconstruction of X or lower dimension)
    Return:
    '''
    if X.shape[1]>300:
        X = X[:,np.random.randint(0,X.shape[1],300)]


    v_obs = []
    v_nul = []
    for i in range(X.shape[1]):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X[:,i], test_size=0.2)
        model = LinearRegression().fit(Z_train, X_train)
        X_pred = model.predict(Z_test)
        v_obs.append(np.less(X_pred,X_test).sum()/len(X_test))
        #v_nul.append(np.less(X_pred,X_train.mean()).sum()/len(X_test))


    n = len(v_obs)
    m, se = np.mean(v_obs), np.std(v_obs)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    if m-h<= 0.5 and 0.5 <= m+h:
        return v_obs, True
    else:
        return v_obs, False

def preprocessing_dg1(name):
    '''
    Pre-Processing driver genes Intogen
    #This first dataset only has information about:
    #BLCA, BRCA ,ESCA,HNSC , LGG, LUSC,PAAD,PRAD
    #missing: ACC, CHOL, LIHC, SARC, SKCM, TCGT, UCS
    Parameters:
        name: file name (str)
    Output:
        csv file
    '''

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

def cgc():
    path = 'data\\cancer_gene_census.csv'
    dgenes = pd.read_csv('data\\cancer_gene_census.csv',sep=',')
    #remove = ['Synonyms',]
    #dgenes = dgenes.dropna(axis=1)
    #print(dgenes.loc[dgenes['Tumour Types(Somatic)'].isna(),'Tumour Types(Somatic)'].shape)
    #dgenes.loc[dgenes['Tumour Types(Somatic)'].isna(),'Tumour Types(Somatic)'] = dgenes['Tumour Types(Somatic)']
    #print(dgenes.loc[dgenes['Tumour Types(Somatic)'].isna(),'Tumour Types(Somatic)'].shape)
    dgenes['Tumour Types(Somatic)'] = dgenes['Tumour Types(Somatic)'].fillna(dgenes['Tumour Types(Germline)'])

    #treat cases where these two values are different
    #starnd names of tumors

    #ct_rawnames = [dgenes.iloc[:,9].unique(),dgenes.iloc[:,10].unique()]
    #ct_rawnames2 = [item for sublist in ct_rawnames for item in sublist]
    #ct_rawnames2 = [str(item).split(", ") for item in ct_rawnames2]
    #ct_rawnames2 = [item for sublist in ct_rawnames2 for item in sublist]
    #ct_rawnames2 = list(set(ct_rawnames2))

    return dgenes#,ct_rawnames2

def outcome_model(train,colnames , z, y01):
    '''
    Outcome Model + logistic regression
    I need to use less features for each model, so i can run several
    batches of the model using the latent features. They should account
    for all cofounders from the hole data
    parameters:
        train: dataset with original features jxv
        z: latent features, jxk
        y01: response, jx1

    return: list of significant coefs
    '''
    col_new_order = []
    col_pvalue = []
    col_coef = []

    if train.shape[1]>100:
        np.random.seed(10)
        columns_split = np.random.randint(0,train.shape[1]//100,train.shape[1] )

    for cs in range(0,train.shape[1]//100):
        cols = np.arange(train.shape[1])[np.equal(columns_split,cs)]
        colnames_sub = colnames[np.equal(columns_split,cs)]
        X = pd.concat([pd.DataFrame(train[:,cols]),pd.DataFrame(z)], axis= 1)
        X.columns = range(0,X.shape[1])
        output = sm.Logit(y01, X).fit()
        col_new_order.extend(colnames_sub)
        col_pvalue.extend(output.pvalues[0:(len(output.pvalues)-z.shape[1])])
        col_coef.extend(output.params[0:(len(output.pvalues)-z.shape[1])])
        if cs==0:
            print('---colname:',colnames_sub[0], ' and pvalue ',output.pvalues[0],'---')


    resul =  pd.concat([pd.DataFrame(col_new_order),pd.DataFrame(col_pvalue), pd.DataFrame(col_coef)], axis = 1)
    resul.columns = ['genes','pvalue','coef']
    return resul, output
