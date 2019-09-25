'''Loading libraries'''
import pandas as pd 
import numpy as np 
#import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#from scipy.stats import gamma
#import copy

'''
Notes:
- DATALIMIT: V = 3500 WITH EVERYTHING ELSE CLOSED 
- compute canada: problem with files location 
'''



'''Hyperparameters'''
k = 30#Latents Dimension 
#sim = 600 #Simulations 
#bach_size = 200 #Batch size for memory purposes 
#step1 = 10 #Saving chain every step1 steps 
#step2 = 20

#WRONG< UPDATE HERE 
#if bach_size//step2 <= 20:
#    print('ERROR ON MCMC, this division must be bigger than 20')

'''Loading dataset'''
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_filtered.txt"
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_binary.txt"
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_gexpression.txt"
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_ge_balanced.txt"
#filename = "C:\\Users\\raque\\Documents\\GitHub\\project_spring2019\\Data\\tcga_train_gexpression.txt"
data = pd.read_csv(filename, sep=';')


f1_sample = []
acc_sample = []

for sample in np.arange(0,100):
    '''Splitting Dataset'''
    train, test = train_test_split(data, test_size=0.3) #random_state=22
    #train = data
    
    '''Organizing columns names'''
    remove = train.columns[[0,1]]
    y = train.columns[1]
    y01 = np.array(train[y])
    train = train.drop(remove, axis = 1)
    y01_t = np.array(test[y])
    test = test.drop(remove, axis = 1)
    
    
    '''Defining variables'''
    v = train.shape[1] #genes
    j = train.shape[0] #patients 
    
    
    '''Matrix Factorization''' 
    from sklearn.decomposition import NMF
    model1 = NMF(n_components=k, init='random') #random_state=0
    latent1 = model1.fit_transform(train)
    latent1_t= model1.transform(test)
    
    '''Logistc Regression''' 
    from sklearn.linear_model import LogisticRegressionCV
    model6 = LogisticRegressionCV(Cs=6,penalty='l2',fit_intercept=True).fit(latent1,y01)
    pred6 = model6.predict(latent1)
    pred6_t = model6.predict(latent1_t)
    
    print(confusion_matrix(pred6,y01))
    ac = confusion_matrix(pred6_t,y01_t)
    print('testing set')
    print((ac[0,0]+ac[1,1])/ac.sum())
    f1_sample.append(f1_score(y01_t,pred6_t))
    acc_sample.append((ac[0,0]+ac[1,1])/ac.sum())
    #with k=100, tcga_train_geexpression, acc on testing set is 0.73 and training set is 0.79



with open('baselines_f1.txt', 'w') as f:
    for item in f1_sample:
        f.write("%s\n" % item)

with open('baselines_acc.txt', 'w') as f:
    for item in acc_sample:
        f.write("%s\n" % item)

print('min, mean, max and var \n')
print('F1: ',f1_sample.min(), f1_sample.mean(),f1_sample.max() ,f1_sample.var(),'\n')
print('Acc: ',acc_sample.min(), acc_sample.mean(),acc_sample.max() ,acc_sample.var())

'''Auto-enconder''' 
#https://github.com/greenelab/tybalt
'''PCA
from sklearn.decomposition import PCA
model3 = PCA(n_components = 100)
latent3 = model3.fit_transform(train)
''' 
'''Random Forest''' 
#mod4 

'''Neural Networks''' 
#m5 

'''Logistc Regression
from sklearn.linear_model import LogisticRegressionCV
model6 = LogisticRegressionCV(Cs=6,penalty='l2',fit_intercept=True).fit(latent1,y01)
pred6 = model6.predict(latent1)
''' 
'''Naive Bayes
from sklearn.naive_bayes import GaussianNB
model7 = GaussianNB()
pred7 = model7.fit(latent1, y01).predict(latent1)''' 