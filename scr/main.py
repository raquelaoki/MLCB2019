'''Loading libraries'''
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix, f1_score
path = 'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019'
sys.path.append(path+'\\scr')
import functions as fc
import plots as pl
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
RUN_PMF = False #to implement
RUN_PCA = False
RUN_A = False#lab computer outside anaconda

RUN_FCI = False

RUN_CREATE_FEATURE_DATASET = False
RUN_EXPERIMENTS = False


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
k_pca = k_ac = 20
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

#Running Factor Analysis Models + Predictive Check + outcome model in all
if RUN_ALL:
    data = pd.read_csv(filename, sep=';')
    #data = data.iloc[0:500, 0:100]
    train, j, v, y01, abr, colnames = fc.data_prep(data)

    #Old factor model combined with the classification model
    fc.fa_mcmc(train,y01,sim,bach_size,step1,k_mcmc,id,RUN_MCMC)
    la_sk,la_cj,lm_tht,lm_phi = fc.load_chain(id,sim,bach_size,j,v,k_mcmc,RUN_LOAD_MCMC)
    if RUN_LOAD_MCMC:
        Z = lm_tht.dot(np.transpose(lm_phi))
        v_pred, test_result = fc.predictive_check_new(train,Z,True)
        if(test_result):
            print('Predictive Check test: PASS')
            fc.outcome_model(train,lm_tht,y01)
        else:
            print('Predictive Check Test: FAIL')

    if RUN_MF:
        W, F = fc.fa_matrixfactorization(train,k_mf,RUN_MF)
        pred = fc.check_save(W,train,colnames, y01,'mf','all', k_mf)
        name = 'mf_'+str(k_mf)+'_lr_all'
        fc.roc_curve_points(pred, y01, name)

    if RUN_PCA:
        pc = fc.fa_pca(train,k_pca,RUN_PCA)
        pred = fc.check_save(pc,train,colnames, y01,'pca', 'all',k_pca)
        name = 'pca_'+str(k_pca)+'_lr_all'
        fc.roc_curve_points(pred, y01, name)

    if RUN_A:
        ac =  fc.fa_a(train,k_ac,RUN_A)
        pred = fc.check_save(ac,train,colnames, y01,'ac','all', k_ac)
        name = 'ac_'+str(k_ac)+'_lr_all'
        fc.roc_curve_points(pred, y01, name)

#Running Factor Analysis + Predictive Check + outcome model
if RUN_:
    files = pd.read_csv('data\\files_names.txt',sep=';')
    # row = files.iloc[0,:]
    for i,row in files.iterrows():
        data_ = pd.read_csv('data\\'+row[0],sep=';')
        train, j, v, y01, abr, colnames = fc.data_prep(data_)
        print("TRAIN INFO: \nshape: ",train.shape, '\nclinical info: ', row[1],row[2])

        if RUN_MF and data_.shape[0]>=100:
            W, F = fc.fa_matrixfactorization(train,k_mf,RUN_MF)
            name = str(row[1])+'_'+str(row[2])
            pred = fc.check_save(W,train,colnames, y01,'mf',name, k_mf)
            name = 'mf_'+str(k_mf)+'_lr_'+name
            fc.roc_curve_points(pred, y01, name)

        if RUN_PCA and data_.shape[0]>=100:
            pc = fc.fa_pca(train,k_pca,RUN_PCA)
            name = str(row[1])+'_'+str(row[2])
            pred = fc.check_save(pc,train,colnames, y01,'pca', name,k_pca)
            name = 'pca_'+str(k_mf)+'_lr_'+name
            fc.roc_curve_points(pred, y01, name)

        if RUN_A and data_.shape[0]>=100:
            ac =  fc.fa_a(train,k_ac,RUN_A)
            name = str(row[1])+'_'+str(row[2])
            pred = fc.check_save(ac,train,colnames, y01,'ac',name, k_ac)
            name = 'ac_'+str(k_mf)+'_lr_'+name
            fc.roc_curve_points(pred, y01, name)

if RUN_FCI:
    from pycausal import search as s
    from pycausal.pycausal import pycausal as pc
    from pycausal import prior as p
    #https://github.com/bd2kccd/py-causal/blob/development/example/py-causal%20-%20GFCI%20Continuous%20in%20Action.ipynb
    pc = pc()
    pc.start_vm()
    #forbid = [['TangibilityCondition','Impact']]
    #require =[['Sympathy','TangibilityCondition']]
    #tempForbid = p.ForbiddenWithin(['TangibilityCondition','Imaginability'])
    #temporal = [tempForbid,['Sympathy','AmountDonated'],['Impact']]
    #prior = p.knowledge(forbiddirect = forbid, requiredirect = require, addtemporal = temporal)
    #prior
    #https://rawgit.com/cmu-phil/tetrad/development/tetrad-gui/src/main/resources/resources/javahelp/manual/tetrad_tutorial.html
    tetrad = s.tetradrunner()

    #what are the best options?
    #tetrad.listIndTests()
    #tetrad.listScores()
    train_p = pd.DataFrame(train[:,0:2000])
    tetrad.getAlgorithmParameters(algoId = 'gfci', testId = 'fisher-z-test', scoreId = 'sem-bic')
    tetrad.run(algoId = 'gfci', dfs = train_p, testId = 'fisher-z-test', scoreId = 'sem-bic',
               maxDegree = 5, maxPathLength = 10,
               completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True)
    #tetrad.getEdges()
    #tetrad.getNodes()
    #pc.stop_vm()

#merge is weird, the sizes are not compatible
#check the indivicual run for subgroup. Any reason to have different dimension for cancer type? Problem on dataset creation?
if RUN_CREATE_FEATURE_DATASET:
    f_bart, f_mf,f_mf_bin , f_ac,f_ac_bin, f_pca,f_pca_bin = fc.data_features_construction(path)
    print(f_bart.shape, f_mf.shape,f_mf_bin.shape,f_ac.shape,f_ac_bin.shape,f_pca.shape,f_pca_bin.shape)

    '''Driver Genes'''
    cgc_list = fc.cgc()
    cgc_list['y_out']=1
    cgc_list = cgc_list.iloc[:,[0,-1]]
    cgc_list.rename(columns = {'Gene Symbol':'genes'}, inplace = True)

    #data_out = pd.merge(f_bart, f_mf, on='genes') #7066 rows
    #data_out = pd.merge(cgc_list,data_out,on='genes',how='right')



if RUN_EXPERIMENTS:
    #Description:
    #All, Gender, Cancer, all+gender, all+cancer, gender + cancer, all+cancer+gender
    name_in = [['all'],['FEMALE','MALE'],[], ['all','FEMALE','MALE'],['all'],['FEMALE','MALE'],[]]
    name_out = [[],[],['all','FEMALE','MALE'],[],['FEMALE','MALE'],['all'],[]]
    aux = True
    for nin, nout in zip(name_in, name_out):
        print(aux,'\n',nin,'\n',nout)
        dt0, dt1, dt2, dt3, dt4, dt5, dt6 = fc.data_subseting(f_bart, f_mf,f_mf_bin , f_ac,f_ac_bin, f_pca,f_pca_bin, nin, nout)

        # do the same with bin
        data_list, names = fc.data_merging(dt0,dt1,dt3, dt5, cgc_list, ['bart','mf','ac','pca'])
        data_list_b, names_b = fc.data_merging(dt0,dt2,dt4, dt6, cgc_list, ['bart_b','mf_b','ac_b','pca_b'])
        if aux:
            dt_exp = fc.data_running_models(data_list, names,nin,nout)
            dt_exp = pd.concat([dt_exp, fc.data_running_models(data_list_b, names_b,nin,nout)], axis=0)
            aux = False
        else:
            dt_exp = pd.concat([dt_exp, fc.data_running_models(data_list, names,nin,nout)], axis=0)
            dt_exp = pd.concat([dt_exp, fc.data_running_models(data_list_b, names_b,nin,nout)], axis=0)
    #dt_exp.to_csv('results\\experiments.txt', sep=';', index = False)

RUN_PLOTS = True
if RUN_PLOTS:
    dt_exp = pd.read_csv('results\\experiments.txt',sep=';')
    dt_exp = dt_exp[dt_exp['error']==False]

    #Comparing for all patients and testing set results for acc and f1
    aux = []
    aux_ = [] #saving columns positions
    columns = ['acc','f1','model_name','data_name','nin','nout']
    for c in columns:
        if c=='acc' or c=='f1':
            aux.append(dt_exp.columns.get_loc(c))
            aux_.append(dt_exp.columns.get_loc(c+'_'))
        else:
            aux.append(dt_exp.columns.get_loc(c))
            aux_.append(dt_exp.columns.get_loc(c))

    dt_exp1 = dt_exp.iloc[:,aux]
    dt_exp1_ = dt_exp.iloc[:,aux_]
    dt_exp1 = dt_exp1.assign(Data='Testing Set')
    dt_exp1_ = dt_exp1_.assign(Data='Full Set')
    dt_exp1_.rename(columns={'acc_':'acc','f1_':'f1','model_name':'Model'},inplace = True)
    dt = pd.concat([dt_exp1,dt_exp1_],axis = 0)
    dt['Model'].replace({'OneClassSVM':'One Class \nSVM','svm':'SVM',
                              'adapter':'PU-Adapter','upu':'Unbiased \nPU',
                              'lr':'Logistic \nRegression','randomforest':'Random \nForest'}, inplace=True)
    #pl.violin_plot(dt)
    import seaborn as sns
    import matplotlib.pyplot as plt
    #plt.figure(figsize=(16, 6))
    sns.set(rc={'figure.figsize':(13.7,8.27)},style="whitegrid",font_scale=2)
    ax = sns.violinplot(x='Model',y='acc',data=dt)
    plt.xlabel('Accuracy')
    plt.show(ax)
    bx = sns.violinplot(x='Model',y='f1',data=dt)
    cx = sns.violinplot(x='Model',y='acc',hue='Data',data=dt, split=True)
    dx = sns.violinplot(x='Model',y='acc',hue='Data',data=dt, split=True)
