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
import warnings
from os import listdir
from os.path import isfile, join
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
    parameters:
        data: full dataset
    return:
        train: training set withouht these elements
        j,v: dimensions
        y01: classifications
        abr: cancer types
        colnames: gene names
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

def fa_pmc(train,k,run):
    #https://stats.stackexchange.com/questions/146547/pymc3-implementation-of-probabilistic-matrix-factorization-pmf-map-produces-a
    #https://gist.github.com/macks22/00a17b1d374dfc267a9a
    import pymc3 as pm
    import theano
    import theano.tensor as t

    """Construct the Probabilistic Matrix Factorization model using pymc3.
    Note that the `testval` param for U and V initialize the model away from
    0 using a small amount of Gaussian noise.
    :param np.ndarray train: Training data (observed) to learn the model on.
    :param int alpha: Fixed precision to use for the rating likelihood function.
    :param int dim: Dimensionality of the model; rank of low-rank approximation.
    :param float std: Standard deviation for Gaussian noise in model initialization.
    """
    # Mean value imputation on training data.
    train = train.copy()
    nan_mask = np.isnan(train)
    train[nan_mask] = train[~nan_mask].mean()

    # Low precision reflects uncertainty; prevents overfitting.
    # We use point estimates from the data to intialize.
    # Set to mean variance across users and items.
    alpha_u = 1 / train.var(axis=1).mean()
    alpha_v = 1 / train.var(axis=0).mean()

    logging.info('building the PMF model')
    n, m = train.shape
    with pm.Model() as pmf:
        U = pm.MvNormal(
            'U', mu=0, tau=alpha_u * np.eye(dim),
            shape=(n, dim), testval=np.random.randn(n, dim) * std)
        V = pm.MvNormal(
            'V', mu=0, tau=alpha_v * np.eye(dim),
            shape=(m, dim), testval=np.random.randn(m, dim) * std)
        R = pm.Normal(
            'R', mu=t.dot(U, V.T), tau=alpha * np.ones(train.shape),
            observed=train)

    logging.info('done building PMF model')
    return pmf

def check_save(Z,train,colnames,y01,name1,name2,k):
    '''
    Run predictive check function and print results
    input:
        z: latent features
        train: training set
        colnames: genes names
        y01: binary classification
        name: name for the file
        k: size of the latent features (repetitive)
    output:
        prints

    '''
    v_pred, test_result = predictive_check_new(train,Z,True)
    if(test_result):
        print('Predictive Check test: PASS')
        resul, output, pred = outcome_model( train,colnames, Z,y01,name2)
        if(len(pred)!=0):
            resul.to_csv('results\\feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt', sep=';', index = False)
        else:
            print('Outcome Model Does Not Coverge, results are not saved')
    else:
        print('Predictive Check Test: FAIL')
        print('Results not saved')
    return pred

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
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X[:,i], test_size=0.3)
        model = LinearRegression().fit(Z_train, X_train)
        X_pred = model.predict(Z_test)
        v_obs.append(np.less(X_test, X_pred).sum()/len(X_test))
        v_nul.append(np.less(X_test, X_train.mean(),).sum()/len(X_test))

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

def outcome_model(train,colnames , z, y01,name2):
    '''
    Outcome Model + logistic regression
    I need to use less features for each model, so i can run several
    batches of the model using the latent features. They should account
    for all cofounders from the hole data

    pred: is the average value thought all the models

    parameters:
        train: dataset with original features jxv
        z: latent features, jxk
        y01: response, jx1

    return: list of significant coefs
    '''
    aux = train.shape[0]//25
    

    lim = 0
    col_new_order = []
    col_pvalue = []
    col_coef = []

    pred = []
    warnings.filterwarnings("ignore")


    #while flag == 0 and lim<=50:
    if train.shape[1]>aux:
        columns_split = np.random.randint(0,train.shape[1]//aux,train.shape[1] )
    
    #print('Aux value: ',aux)
    flag1 = 0
    for cs in range(0,train.shape[1]//aux):
        
        cols = np.arange(train.shape[1])[np.equal(columns_split,cs)]
        colnames_sub = colnames[np.equal(columns_split,cs)]
        col_new_order.extend(colnames_sub)
        X = pd.concat([pd.DataFrame(train[:,cols]),pd.DataFrame(z)], axis= 1)
        X.columns = range(0,X.shape[1])
        flag = 0 
        lim = 0 
        while flag==0 and lim <= 50:
            try:
                output = sm.Logit(y01, X).fit(disp=0)
                #if cs==0:
                #    print('---colname:',colnames_sub[0], ' and pvalue ',output.pvalues[0],'---')
                #Predictions
                pred.append(output.predict(X))
                flag = 1
                flag1 = 1
            except:
                flag = 0
                lim = lim+1
                print('--------- Trying again----------- ',name2, aux,cs)

        if flag == 1: 
            col_pvalue.extend(output.pvalues[0:(len(output.pvalues)-z.shape[1])])
            col_coef.extend(output.params[0:(len(output.pvalues)-z.shape[1])]) 
        else: 
            col_pvalue.extend(np.repeat(0,len(colnames_sub)))
            col_coef.extend(np.repeat(0,len(colnames_sub))) 

    warnings.filterwarnings("default")
    #prediction only for the ones with models that converge
    pred1 =  np.mean(pred,axis = 0)
    resul =  pd.concat([pd.DataFrame(col_new_order),pd.DataFrame(col_pvalue), pd.DataFrame(col_coef)], axis = 1)
    resul.columns = ['genes','pvalue','coef']
    if flag1 == 0: 
        output = []
        pred1 = []
    return resul, output, pred1



#todo: change outcome functions to return pred values
def roc_curve_points(pred,y01,name):
    '''
    Saving points to create a roc curve
    Input:
        pred: predictive values (from outcome model, probability to belong a class)
        y01: true binary value (observed)
        name: for saving Values
    Output: dataframe on results folder
        prob: threshold
        tp1:  0,0/(0,0+0,1)
        fp1:  1,0/(1,0+1,1)
        tp2:  1,1/(1,1+1,0)
        fp2: 0,1/(0,1+0,0)


      tab = table(tp,obs)
      #Prob, opt1_tp,opt1_fp,opt2_tp,opt2_fp
      output = c(prob,tab[1,1]/(tab[1,1]+tab[1,2]),tab[2,1]/(tab[2,1]+tab[2,2]),
                      tab[2,2]/(tab[2,2]+tab[2,1]), tab[1,2]/(tab[1,2]+tab[1,1]))
      return(output)
    }
    roc_data = rbind(def_roc(pred_p,y,0.01),def_roc(pred_p,y,0.02))
    values = seq(0.03,1,by=0.01)
    for(i in 1:length(values)){
      roc_data = rbind(roc_data,def_roc(pred_p,y,values[i]))
    }
    roc_data = data.frame(roc_data)
    names(roc_data) = c('prob','tp1','fp1','tp2','fp2')
    write.table(roc_data,'results\\roc_bart_all.txt', row.names = FALSE,sep=';')
    '''
   
    
    warnings.filterwarnings("ignore")
    if len(pred)>0:
        seq = np.arange(0.01,1,step = 0.01)
        tp1 , tp2, fp1 , fp2 = [],[],[],[]
        for s in seq:
            tp = copy.deepcopy(pred)  #1-pred
            tp[tp<=s] = 0
            tp[tp>s] = 1
    
            tn,fp,fn,tp = confusion_matrix(y01,tp).ravel()
            tp1.append(tn/(tn+fn))
            fp1.append(fp/(fp+tp))
            tp2.append(tp/(tp+fp))
            fp2.append(fn/(tn+fn))
        roc_data = pd.DataFrame({'prob':seq, 'tp1':tp1,'fp1':fp1 ,'tp2':tp2,'fp2':fp2})
        roc_data.to_csv('results\\roc_'+name+'.txt', sep=';', index = False)
        warnings.filterwarnings("default")

    return

def data_features_da_create(data,files):
    #create 2 dataset
    data['aux'] = np.where(data['pvalue']<0.05,1,0)
    features_data_bin = data.iloc[:,[0,-1]]
    features_data_bin.rename(columns={ 'aux':files[-1].split('.')[0]}, inplace = True)
 
    features_data = data.iloc[:,[0,2]]
    features_data.rename(columns={ 'coef':files[-1].split('.')[0]}, inplace = True)
    if data.shape[0]!=features_data.shape[0] or data.shape[0]!=features_data_bin.shape[0]:
        print('Dimension Problem with ', files)
    return  features_data,features_data_bin

                
def data_features_da_merge(features_data, features_data_bin, data, files):
    #create binary variable 
    data['aux'] = np.where(data['pvalue']<0.05,1,0)
    #merge data with current dataset with binary data
    features_data_bin = pd.merge(features_data_bin, data.iloc[:,[0,-1]],on='genes')
    features_data_bin.rename(columns={ 'aux':files[-1].split('.')[0]}, inplace = True)
    #merge data with current dataset with continuos data
    features_data= pd.merge(features_data, data.iloc[:,[0,2]],on='genes')
    features_data.rename(columns={ 'coef':files[-1].split('.')[0]}, inplace = True)  

    if data.shape[0]!=features_data.shape[0] or data.shape[0]!=features_data_bin.shape[0]:
        print('Dimension Problem with ', files, data.shape[0],features_data.shape[0])

    return  features_data,features_data_bin

def data_features_construction(path):
    '''
    This function will read the features in results and
    construct 2 datasets: one with the data for the ROC curve
    and another one with the features for the data out/classification
    '''
    pathfiles = path+'\\results'
    listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
    flags = {'pca': True, 'mf': True , 'ac':True,'bart':True ,'roc':True} #flags['pca']
    
    for f in listfiles:
        data = pd.read_csv('results//'+f,sep=';')
        files = f.split("_")
        if files[0]=="feature":
            
            if files[1]=='bart' and flags['bart']:
                #Create dataset 
                features_bart =  data.iloc[:,0:2]
                features_bart.rename(columns={ 'mean':files[-1].split('.')[0]}, inplace = True)
                flags['bart'] = False   
            elif files[1]=='bart' and not flags['bart']:
                #merge with dataset
                features_bart = pd.merge(features_bart, data.iloc[:,0:2],on='gene')
                features_bart.rename(columns={ 'mean':files[-1].split('.')[0]}, inplace = True)
                if data.shape[0]!=features_bart.shape[0]:
                    print('Dimension Problem with ', files)

                
            elif files[1]=='mf' and flags['mf']:
                features_mf,features_mf_binary = data_features_da_create(data,files)
                flags['mf'] = False  
            elif files[1]=='mf' and not flags['mf']:
                features_mf,features_mf_binary = data_features_da_merge(features_mf,features_mf_binary,data, files)
                
            elif files[1]=='ac' and flags['ac']:
                features_ac,features_ac_binary = data_features_da_create(data,files)
                flags['ac'] = False  
            elif files[1]=='ac' and not flags['ac']:
                features_ac,features_ac_binary = data_features_da_merge(features_ac,features_ac_binary,data,files)

            elif files[1]=='pca' and flags['pca']:
                features_pca,features_pca_binary = data_features_da_create(data,files)
                flags['pca'] = False  
            elif files[1]=='pca' and not flags['pca']:
                features_pca,features_pca_binary = data_features_da_merge(features_pca,features_pca_binary,data,files)
            else: 
                print('ERROR: ', f)
            
        else:
            print('roc:', files[0])
            #add to previous roc dataset

    return features_bart, features_mf,features_mf_binary , features_ac,features_ac_binary , features_pca,features_pca_binary














