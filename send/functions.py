import os
import time
import copy
import numpy as np
import pandas as pd
import warnings
from itertools import compress

from os import listdir
from os.path import isfile, join

from sklearn import svm
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator

from scipy.stats import gamma
from scipy import sparse, stats
import statsmodels.discrete.discrete_model as sm

from puLearning.puAdapter import PUAdapter
from pywsl.pul import pumil_mr, pu_mr
from pywsl.utils.syndata import gen_twonorm_pumil
from pywsl.utils.comcalc import bin_clf_err

'''PART 0: Data Preparation'''

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

    j, v = train.shape

    return train, j, v, y01,  abr, colnames

'''PART 1: DECONFOUNDER ALGORITHM'''

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
        save features on results folder or print that it failed
        return the predicted values for the training data on the outcome model
    '''

    v_pred, test_result = predictive_check_new(train,Z,True)
    if(test_result):
        print('Predictive Check test: PASS')
        resul, output, pred = outcome_model( train,colnames, Z,y01,name2)
        if(len(pred)!=0):
            resul.to_csv('results\\feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt', sep=';', index = False)
            name = name1+str(k)+'_lr_'+name2
            roc_curve_points(pred, y01, name)
        else:
            print('Outcome Model Does Not Coverge, results are not saved')
            empty = []
            np.savetxt('results\\FAIL_outcome_feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt',[], fmt='%s')

    else:
        print('Predictive Check Test: FAIL')
        np.savetxt('results\\FAIL_pcheck_feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt',[], fmt='%s')
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
        v_obs values and result of the test
    '''

    #If the number of columns is too large, select a subset of columns instead
    if X.shape[1]>10000:
        X = X[:,np.random.randint(0,X.shape[1],10000)]

    v_obs = []
    v_nul = []
    for i in range(X.shape[1]):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X[:,i], test_size=0.3)
        model = LinearRegression().fit(Z_train, X_train)
        X_pred = model.predict(Z_test)
        v_obs.append(np.less(X_test, X_pred).sum()/len(X_test))
        v_nul.append(np.less(X_test, X_train.mean(),).sum()/len(X_test))

    #Create the Confidence interval
    n = len(v_nul)
    m, se = np.mean(v_nul), np.std(v_nul)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    if m-h<= np.mean(v_obs) and np.mean(v_obs) <= m+h:
        return v_obs, True
    else:
        return v_obs, False

def preprocessing_dg1(name):
    '''
    Pre-Processing driver genes
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
    dgenes['Tumour Types(Somatic)'] = dgenes['Tumour Types(Somatic)'].fillna(dgenes['Tumour Types(Germline)'])
    return dgenes

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
    #if ac, change 25 to 9
    aux = train.shape[0]//25


    lim = 0
    col_new_order = []
    col_pvalue = []
    col_coef = []

    pred = []
    warnings.filterwarnings("ignore")

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

def roc_curve_points(pred,y01,name):
    roc_data = pd.DataFrame({'pred':pred,'y01':y01})
    roc_data.to_csv('results\\roc_'+name+'.txt', sep=';', index = False)

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
        data = pd.read_csv('results\\'+f,sep=';')
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
    features_bart.rename(columns={'gene':'genes'}, inplace = True)
    return features_bart, features_mf,features_mf_binary , features_ac,features_ac_binary , features_pca,features_pca_binary

def pul(y,y_test,X,X_test,aux,name_model):
    """
    Input:
        X,y,X_test, y_test: dataset to train the model
        aux: output to save the roc curve (not currently implemented)
    Return:
        cm: confusion matrix for the testing set
        cm_: confusion matrix for the full dataset
        y_all_: prediction for the full dataset
    """
    X_full = np.concatenate((X,X_test), axis = 0 )
    y_full = np.concatenate((y,y_test), axis = 0 )
    warnings.filterwarnings("ignore")
    if name_model == 'OneClassSVM':
        #modify dataset to have only positive examples on testing set
        X = X[y==1]
        print('OneClassSVM',X.shape[1])
        model = svm.OneClassSVM(nu=0.1,kernel="rbf",gamma=0.5)# #0.5, 0.5
        model.fit(X)


    elif name_model == 'svm':
        #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        print('svm',X.shape[1])
        model = svm.SVC(C=0.5,kernel='rbf',gamma='scale') #overfitting
        model.fit(X,y)

    elif name_model == 'adapter':
        print('adapter',X.shape[1])
        estimator = SVC(C=0.3, kernel='rbf',gamma='scale',probability=True)
        model = PUAdapter(estimator, hold_out_ratio=0.3)
        X = np.matrix(X)
        y = np.array(y)
        model.fit(X, y)

    elif name_model == 'upu':
        '''
        pul: nnpu (Non-negative PU Learning), pu_skc(PU Set Kernel Classifier),
        pnu_mr:PNU classification and PNU-AUC optimization (the one tht works: also use negative data)
        nnpu is more complicated (neural nets, other methos seems to be easier)
        try https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/pu_skc/demo_pu_skc.py
        and https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
         '''
        print('upu', X.shape[1])
        #Implement these, packages only work on base anaconda (as autoenconder)
        #https://github.com/t-sakai-kure/pywsl
        prior =.5 #change for the proportion of 1 and 0
        param_grid = {'prior': [prior],
                          'lam': np.logspace(-3, 1, 5), #what are these values
                          'basis': ['lm']}
        lambda_list = np.logspace(-3, 1, 5)
        #upu (Unbiased PU learning)
        #https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
        model = GridSearchCV(estimator=pu_mr.PU_SL(),
                               param_grid=param_grid, cv=10, n_jobs=-1)
        X = np.matrix(X)
        y = np.array(y)
        model.fit(X, y)
    elif name_model == 'lr':
        print('lr',X.shape[1])
        model = sm.Logit(y,X).fit_regularized(method='l1')
    elif name_model=='randomforest':
        print('rd',X.shape[1])
        md = max(np.floor(X.shape[1]/3),6)
        model = RandomForestClassifier(max_depth=md, random_state=0)
        model.fit(X, y)
    else:
        print('random',X.shape[1])


    #I changed here
    if name_model=='random':
        y_ = np.random.binomial(n=1,p=y.sum()/len(y),size =X_test.shape[0])
        y_full_ = np.random.binomial(n=1,p=y.sum()/len(y),size=X_full.shape[0])
    else:
        y_ = model.predict(X_test)
        y_full_ = model.predict(X_full)

    if name_model == 'lr':
        y_[y_<0.5] = 0
        y_[y_>=0.5] = 1
        y_full_[y_full_< 0.5] = 0
        y_full_[y_full_>=0.5] = 1

    y_ = np.where(y_==-1,0,y_)
    y_full_ = np.where(y_full_==-1, 0,y_full_)

    acc = accuracy_score(y_test,y_)
    acc_f = accuracy_score(y_full, y_full_)
    f1 = f1_score(y_test,y_)
    f1_f = f1_score(y_full, y_full_)
    tnfpfntp = confusion_matrix(y_test,y_).ravel()
    tnfpfntp_= confusion_matrix(y_full, y_full_).ravel()
    tp_genes = np.multiply(y_full, y_full_)
    warnings.filterwarnings("default")
    return [acc, acc_f, f1, f1_f], tnfpfntp, tnfpfntp_, tp_genes,y_,y_full_

def data_subseting(data0, data1, data2, data3, data4, data5, data6, name_in, name_out):
    '''
    Select the features that I want to work with:
    all: based on all cancer patients
    gender: features from causal models on males and female
    abr: features from causal models on subcancer types
    '''
    if len(name_out)>0:
        data0 = data0.drop(columns=name_out)
        data1 = data1.drop(columns=name_out)
        data2 = data2.drop(columns=name_out)
        data3 = data3.drop(columns=name_out)
        data4 = data4.drop(columns=name_out)
        data5 = data5.drop(columns=name_out)
        data6 = data6.drop(columns=name_out)

    if len(name_in)>0:
        aux0, aux1, aux2, aux3, aux4, aux5, aux6 = [0],[0], [0], [0], [0], [0], [0]
        for i in name_in:
            aux0.append(data0.columns.get_loc(i))
            aux1.append(data1.columns.get_loc(i))
            aux2.append(data2.columns.get_loc(i))
            aux3.append(data3.columns.get_loc(i))
            aux4.append(data4.columns.get_loc(i))
            aux5.append(data5.columns.get_loc(i))
            aux6.append(data6.columns.get_loc(i))
        data0 = data0.iloc[:,aux0]
        data1 = data1.iloc[:,aux1]
        data2 = data2.iloc[:,aux2]
        data3 = data3.iloc[:,aux3]
        data4 = data4.iloc[:,aux4]
        data5 = data5.iloc[:,aux5]
        data6 = data6.iloc[:,aux6]

    return data0, data1, data2, data3, data4, data5, data6

def data_merging(data0,data1,data2,data3, cgc, data_names):
    '''
    Merge different datasets and the cgc list
    '''
    #One datset with features only
    d0 = pd.merge(cgc, data0, on='genes',how='right')
    d1 = pd.merge(cgc, data1, on='genes',how='right')
    d2 = pd.merge(cgc, data2, on='genes',how='right')
    d3 = pd.merge(cgc, data3, on='genes',how='right')
    data_names_list = data_names

    #Two datsets with features
    d4 = cgc.merge(data0, on='genes',how='right').merge(data1,on='genes')
    d5 = cgc.merge(data0, on='genes',how='right').merge(data2,on='genes')
    d6 = cgc.merge(data0, on='genes',how='right').merge(data3,on='genes')
    d7 = cgc.merge(data1, on='genes',how='right').merge(data2,on='genes')
    d8 = cgc.merge(data1, on='genes',how='right').merge(data3,on='genes')
    d9 = cgc.merge(data2, on='genes',how='right').merge(data3,on='genes')
    data_names_list.append(data_names[0]+'_'+data_names[1])
    data_names_list.append(data_names[0]+'_'+data_names[2])
    data_names_list.append(data_names[0]+'_'+data_names[3])
    data_names_list.append(data_names[1]+'_'+data_names[2])
    data_names_list.append(data_names[1]+'_'+data_names[3])
    data_names_list.append(data_names[2]+'_'+data_names[3])

    #Three datsets with features
    d10 = pd.merge(d4, data2, on='genes')
    d11 = pd.merge(d4, data3, on='genes')
    d12 = pd.merge(d7, data3, on='genes')
    data_names_list.append(data_names[0]+'_'+data_names[1]+'_'+data_names[2])
    data_names_list.append(data_names[0]+'_'+data_names[1]+'_'+data_names[3])
    data_names_list.append(data_names[1]+'_'+data_names[2]+'_'+data_names[3])    #Four datasets with features
    d13 = pd.merge(d10,data3, on='genes')
    data_names_list.append(data_names[0]+'_'+data_names[1]+'_'+data_names[2]+'_'+data_names[3])

    d0.set_index('genes',inplace=True)
    d1.set_index('genes',inplace=True)
    d2.set_index('genes',inplace=True)
    d3.set_index('genes',inplace=True)
    d4.set_index('genes',inplace=True)
    d5.set_index('genes',inplace=True)
    d6.set_index('genes',inplace=True)
    d7.set_index('genes',inplace=True)
    d8.set_index('genes',inplace=True)
    d9.set_index('genes',inplace=True)
    d10.set_index('genes',inplace=True)
    d11.set_index('genes',inplace=True)
    d12.set_index('genes',inplace=True)
    d13.set_index('genes',inplace=True)
    #print('subsets size: \n')
    #print(d0.shape,d1.shape,d2.shape,d3.shape,d4.shape,d5.shape,d6.shape,d7.shape,d8.shape,d9.shape,d10.shape,d11.shape,d12.shape,d13.shape)
    return [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13], data_names_list

def data_running_models(data_list, names, name_in, name_out, is_bin, id):
    '''
    Run all the pu models for the combination of datsets
    input: list with combinations of features and the names of the datsets
    outout:
    '''
    acc_ , acc = [] , []
    f1_, f1 = [],[]
    tnfpfntp,tnfpfntp_ = [],[] #confusion_matrix().ravel()
    tp_genes = []
    model_name, data_name = [],[]
    nin, nout = [],[]
    error = []
    size = []
    id_name = []
    models = ['OneClassSVM','adapter','upu','lr','randomforest','random']
    for dt,dtn in zip(data_list,names):
        if dt.shape[1]>2:
            #print('type: ',dt,dtn,'shape:', dt.shape[1], dt.head())
            #dt['y_out'].fillna(0,inplace = True)
            y = dt['y_out'].fillna(0)
            X = dt.drop(['y_out'], axis=1)
            index_save = X.index
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X = pd.DataFrame(X,index=index_save)
            y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3)
            index_ = [list(X_train.index),list(X_test.index)]
            flat_index = [item for sublist in index_ for item in sublist]
            flat_index = np.array(flat_index)
            e_full_ = np.where(y==1,0,0)
            e_ = np.where(y_test==1,0,0)
            ensemble_c = 0
            for m in models:
                try:
                    scores, cm, cm_, tp_genes01, y_,y_full_ = pul(y_train, y_test, X_train, X_test,'name',m)
                    acc.append(scores[0])
                    acc_.append(scores[1])
                    f1.append(scores[2])
                    f1_.append(scores[3])
                    tnfpfntp.append(cm)
                    tnfpfntp_.append(cm_)
                    tp_genes.append(flat_index[np.equal(tp_genes01,1)])
                    model_name.append(m)
                    data_name.append(dtn)
                    nin.append(name_in)
                    nout.append(name_out)
                    error.append(False)
                    if(m=='adapter' or m=='upu' or m=='lr' or m=='randomforest'):
                        e_full_ = e_full_+y_full_
                        e_ = e_+y_
                        ensemble_c = ensemble_c+1
                except:
                    acc.append(np.nan)
                    acc_.append(np.nan)
                    f1.append(np.nan)
                    f1_.append(np.nan)
                    tnfpfntp.append([np.nan,np.nan,np.nan,np.nan])
                    tnfpfntp_.append([np.nan,np.nan,np.nan,np.nan])
                    tp_genes.append([])
                    model_name.append(m)
                    data_name.append(dtn)
                    nin.append(name_in)
                    nout.append(name_out)
                    error.append(True)
                    print('Error in PUL model',m,dtn)
                size.append(X_train.shape[1])
                id_name.append(id)

            #print('test',ensemble_c,acc)
            e_full_ = np.multiply(e_full_,1/ensemble_c)
            e_ = np.multiply(e_,1/ensemble_c)
            e_full_ = np.where(np.array(e_full_)>0.5,1,0)
            e_ = np.where(np.array(e_)>0.5,1,0)
            y_full = np.concatenate((y_train,y_test), axis = 0 )
            acc.append(accuracy_score(y_test,e_))
            acc_.append(accuracy_score(y_full,e_full_))
            f1.append(f1_score(y_test,e_))
            f1_.append(f1_score(y_full,e_full_))
            tnfpfntp.append(confusion_matrix(y_test,e_).ravel())
            tnfpfntp_.append(confusion_matrix(y_full,e_full_).ravel())
            tp_genes.append([])
            model_name.append('ensemble')
            data_name.append(dtn)
            nin.append(name_in)
            nout.append(name_out)
            error.append(False)
            size.append(X_train.shape[1])
            id_name.append(id)
        else:
            print(dtn, 'only one columns')
    dt_exp = pd.DataFrame({'acc':acc,'acc_':acc_, 'f1':f1, 'f1_':f1_,
                               'tnfpfntp':tnfpfntp, 'tnfpfntp_':tnfpfntp_,
                               'tp_genes':tp_genes,'model_name':model_name , 'data_name':data_name,
                               'nin':nin, 'nout':nout, 'error':error, 'size': size,
                               'id':id_name})
    return dt_exp
