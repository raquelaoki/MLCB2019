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
#testing
#from bartpy.sklearnmodel import SklearnModel
#from joblib import Parallel
#https://github.com/aldro61/pu-learning

 


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

RUN_CREATE_FEATURE_DATASET = True


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
        
    #'''Driver Genes'''
    cgc_list = fc.cgc()
    cgc_list['y_out']=1
    cgc_list = cgc_list.iloc[:,[0,-1]]
    cgc_list.rename(columns = {'Gene Symbol':'genes'}, inplace = True)
    
    #naive classifier 
    f_bart.rename(columns={'gene':'genes'}, inplace = True)
    
    #Here i need to add the method on the column names 
    #Or before when im constructing the features set   
    data_out = pd.merge(f_bart, f_mf, on='genes') #7066 rows
    data_out = pd.merge(cgc_list,data_out,on='genes',how='right')
    data_out['y_out'].fillna(0,inplace = True)
    
    data_out['y_out'].value_counts()
    data_out.set_index('genes',inplace=True)


from sklearn import svm
from sklearn.model_selection import train_test_split

'''SVM'''
y = data_out['y_out']
X = data_out.iloc[:,1:data_out.shape[1]]
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3)
#try other kernels
clf = svm.SVC(C=10,kernel='rbf',gamma='scale') #overfitting 
clf.fit(X_train, y_train)
y_= clf.predict(X_test)
confusion_matrix(y_test, y_)
f1_score(y_test,y_)

'''Logistic Regression'''
import statsmodels.discrete.discrete_model as sm
clf1 = sm.Logit(y_train,X_train).fit()
y_ = clf1.predict(X_test)
y_[y_<0.5]=0
y_[y_>=0.5] = 1
confusion_matrix(y_test, y_)

'''PU Adapter'''
from puLearning.puAdapter import PUAdapter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

y = data_out['y_out']
X = data_out.iloc[:,1:data_out.shape[1]]
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3)
X_train = np.matrix(X_train)
y_train = np.array(y_train)

estimator = SVC(C=10, kernel='rbf',gamma=0.4,probability=True)
pu_estimator = PUAdapter(estimator, hold_out_ratio=0.2)
#X_train.reset_index(drop=True,inplace=True)    
pu_estimator.fit(X_train, y_train)
    
print(pu_estimator)
print("Comparison of estimator and PUAdapter(estimator):")
print("Number of disagreements: ", len(np.where((pu_estimator.predict(X_test) == estimator.predict(X_test)) == False)[0]))
print("Number of agreements: ", len(np.where((pu_estimator.predict(X_test) == estimator.predict(X_test)) == True)[0]))

print("Number of disagreements: ", len(np.where((pu_estimator.predict(X_test) == y_test) == False)[0]))
print("Number of agreements: ", len(np.where((pu_estimator.predict(X_test) == y_test) == True)[0]))

#if __name__ == "__main__":
#    v1,v2 = main()y_= clf.predict([[2., 2.]])

#Implement these
#https://github.com/t-sakai-kure/pywsl

