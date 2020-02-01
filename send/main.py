'''Loading libraries'''
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix, f1_score
path = '~\\Documents\\GitHub\\project'
sys.path.append(path+'\\scr')
import functions as fc
os.chdir(path)
pd.set_option('display.max_columns', 500)


'''
Flags
'''
RUN_MCMC = False
RUN_LOAD_MCMC = False

RUN_ALL = False
RUN_ = False

RUN_MF = False
RUN_PCA = False
RUN_A = False #outside anaconda

RUN_CREATE_FEATURE_DATASET = True
RUN_CREATE_ROC_CAUSAL_DATASET = False
RUN_EXPERIMENTS = True #RUN_CREATE_FEATURE_DATASET also needs to be true


'''MCMC Hyperparameters'''
k_mf_ = [40] #Latents Dimension
k_pca_ = [40]
k_ac_ = [10]

#def main():
'''Loading dataset'''
filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2


#Running Factor Analysis Models + Predictive Check + outcome model in all
if RUN_ALL:
    data = pd.read_csv(filename, sep=';')
    #data = data.iloc[0:500, 0:100]
    train, j, v, y01, abr, colnames = fc.data_prep(data)

    if RUN_MF:
        for k_mf in k_mf_:
            W, F = fc.fa_matrixfactorization(train,k_mf,RUN_MF)
            pred = fc.check_save(W,train,colnames, y01,'mf','all', k_mf)

    if RUN_PCA:
        for k_pca in k_pca_:
            pc = fc.fa_pca(train,k_pca,RUN_PCA)
            pred = fc.check_save(pc,train,colnames, y01,'pca', 'all',k_pca)

    if RUN_A:
        for k_ac in k_ac_:
            ac =  fc.fa_a(train,k_ac,RUN_A)
            pred = fc.check_save(ac,train,colnames, y01,'ac','all', k_ac)

#Running Factor Analysis + Predictive Check + outcome model
if RUN_:
    files = pd.read_csv('data\\files_names.txt',sep=';')
    # row = files.iloc[0,:]
    for i,row in files.iterrows():
        data_ = pd.read_csv('data\\'+row[0],sep=';')
        train, j, v, y01, abr, colnames = fc.data_prep(data_)
        print("TRAIN INFO: \nshape: ",train.shape, '\nclinical info: ', row[1],row[2])

        if RUN_MF and data_.shape[0]>=100:
            for k_mf in k_mf_:
                W, F = fc.fa_matrixfactorization(train,k_mf,RUN_MF)
                name = str(row[1])+'_'+str(row[2])
                pred = fc.check_save(W,train,colnames, y01,'mf',name, k_mf)

        if RUN_PCA and data_.shape[0]>=100:
            for k_pca in k_pca_:
                pc = fc.fa_pca(train,k_pca,RUN_PCA)
                name = str(row[1])+'_'+str(row[2])
                pred = fc.check_save(pc,train,colnames, y01,'pca', name,k_pca)

        if RUN_A and data_.shape[0]>=100:
            for k_ac in k_ac_:
                ac =  fc.fa_a(train,k_ac,RUN_A)
                name = str(row[1])+'_'+str(row[2])
                pred = fc.check_save(ac,train,colnames, y01,'ac',name, k_ac)


if RUN_CREATE_FEATURE_DATASET:
    f_bart, f_mf,f_mf_bin , f_ac,f_ac_bin, f_pca,f_pca_bin = fc.data_features_construction(path)
    print(f_bart.shape, f_mf.shape,f_mf_bin.shape,f_ac.shape,f_ac_bin.shape,f_pca.shape,f_pca_bin.shape)

    '''Driver Genes'''
    cgc_list = fc.cgc()
    cgc_list['y_out']=1
    cgc_list = cgc_list.iloc[:,[0,-1]]
    cgc_list.rename(columns = {'Gene Symbol':'genes'}, inplace = True)


if RUN_CREATE_ROC_CAUSAL_DATASET:
    roc = fc.data_roc_construction(path)

if RUN_EXPERIMENTS:
    #Description:
    #All, Gender, Cancer, all+gender, all+cancer, gender + cancer, all+cancer+gender
    name_in = [['all'],['FEMALE','MALE'],[], ['all','FEMALE','MALE'],[],['FEMALE','MALE'],[]]
    name_out = [[],[],['all','FEMALE','MALE'],[],['FEMALE','MALE'],['all'],[]]
    name_index = [0,1,2,3,4,5,6]
    aux = True
    #nin = name_in[0]
    #nout = name_out[0]
    #id1 = name_index[0]
    for nin, nout, id1 in zip(name_in, name_out, name_index):
        print(aux,'\n',nin,'\n',nout)
        dt0, dt1, dt2, dt3, dt4, dt5, dt6 = fc.data_subseting(f_bart, f_mf,f_mf_bin , f_ac,f_ac_bin, f_pca,f_pca_bin, nin, nout)
        #Not using the binary version
        data_list, names = fc.data_merging(dt0,dt1,dt3, dt5, cgc_list, ['bart','mf','ac','pca'])
        if aux:
            dt_exp = fc.data_running_models(data_list, names,nin,nout,False,id1)
            aux = False
        else:
            dt_exp = pd.concat([dt_exp, fc.data_running_models(data_list, names,nin,nout,False,id1)], axis=0)
            dt_exp.to_csv('results\\experiments2.txt', sep=';', index = False)
