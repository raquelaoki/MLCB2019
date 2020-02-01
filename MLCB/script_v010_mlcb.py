'''Loading libraries'''
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score
from scipy.stats import gamma
import copy
from sklearn.metrics.pairwise import cosine_similarity


'''
Notes:
- Datalimit V = 3500
- 1000sim is 8h
- k = 30
'''

'''Hyperparameters'''
k = 30 #Latents Dimension
sim = 1000 #Simulations
bach_size = 200 #Batch size for memory purposes
step1 = 10 #Saving chain every step1 steps
id = '12' #identification of simulation


'''Loading dataset'''
filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_gexpression.txt"
#filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\tcga_train_ge_balanced.txt"

data = pd.read_csv(filename, sep=';')
#data = data.iloc[:, 0:300]

'''Parameters'''
class parameters:
    __slots__ = ( 'la_cj','la_sk','la_ev','lm_phi','lm_tht')
    def __init__(self,latent_cj,latent_sk, latent_ev,latent_phi ,latent_tht):
        self.la_cj = latent_cj #string of array J
        self.la_sk = latent_sk #matrix Kx2
        self.la_ev = latent_ev #string of  array V
        self.lm_phi = latent_phi #string of matrix (kv) in array format
        self.lm_tht = latent_tht #string of matrix  (jk) in array format



'''Gibbs Sampling'''
def gibbs(current,train0,j,v,k,y01):
    new = copy.deepcopy(current)
    lvjk = np.zeros((v,j,k))

    for ki in np.arange(k):
        #0.79 decrease, 0.8 increase   #0.795 it's perferct for k=100
        lvjk[:,:,ki] = np.dot(0.795*current.lm_phi[:,ki].reshape(v,1), current.lm_tht[:,ki].reshape(1,j))

    lvk = np.random.poisson(lvjk.sum(axis=1))
    ljk = np.random.poisson(lvjk.sum(axis=0))
    for ki in np.arange(k):
        new.lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+current.la_ev),size = 1)
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

'''Accuracy'''
def acc(theta,sk1,cj,y):
    y0 = gamma.logpdf(x=theta,a = sk1[0,:],scale = 1)
    y1 = gamma.logpdf(x=theta,a = sk1[1,:],scale = 1)
    y3 = y1-y0
    y3 = y3.sum(axis=1)
    y3[y3<=0] = 0
    y3[y3>0] = 1
    return confusion_matrix(y,y3)

'''Accuracy'''
def PGM_pred(theta,sk1,cj,y):
    y0 = gamma.logpdf(x=theta,a = sk1[0,:],scale = 1)
    y1 = gamma.logpdf(x=theta,a = sk1[1,:],scale = 1)
    y3 = y1-y0
    y3 = y3.sum(axis=1)
    y3[y3<=0] = 0
    y3[y3>0] = 1
    return y3


f1_sample = []
acc_sample = []
for experiment in np.arange(0,100):
    print('Experiment ', experiment, ' of 100')
    '''Splitting Dataset'''
    train, test = train_test_split(data, test_size=0.3) #random_state=22

    '''Organizing columns names'''
    remove = train.columns[[0,1]]
    y = train.columns[1]
    y01 = np.array(train[y])
    train = train.drop(remove, axis = 1)
    y01_t = np.array(test[y])
    test = test.drop(remove, axis = 1)
    train0 = np.matrix(train)


    '''Defining variables'''
    v = train.shape[1] #genes
    j = train.shape[0] #patients


    '''Initial Values'''
    current = parameters(np.repeat(0.5,j), #la_cj 0.25
                       np.repeat(150.5,k*2).reshape(2,k), #la_sk 62
                       np.repeat(1.0004,v), #la_ev FIXED
                       np.repeat(1/v,v*k).reshape(v,k),#lm_phi v x k
                       np.repeat(150.5,j*k).reshape(j,k)) #lm_theta k x j
                       #np.repeat(0.5, j), #la_pj
                       #np.repeat(0.5,k)) #la_qk


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
        #.print('it should be 981',data.shape)
        for i in np.arange(1,bach_size):
            new  = gibbs(current,train0,j,v,k,y01)
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
                    print(test1.mean(), train0.mean())

            current= copy.deepcopy(new )
        np.savetxt('Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_sk, delimiter=',',fmt='%5s')
        np.savetxt('Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', chain_la_cj, delimiter=',',fmt='%5s')
        np.savetxt('Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_tht, delimiter=',',fmt='%5s')
        np.savetxt('Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', chain_lm_phi, delimiter=',',fmt='%5s')


    print("--- %s min ---" % int((time.time() - start_time)/60))
    print("--- %s hours ---" % int((time.time() - start_time)/(60*60)))


    '''Loading average values back for predictions'''
    ite0 = 2
    la_sk = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lask_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    la_cj = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lacj_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    lm_phi = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmphi_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)
    lm_tht = np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmtht_id'+str(id)+'_bach'+str(ite0)+'.txt', delimiter=',').mean(axis=1)


    for ite in np.arange(ite0+1,sim//bach_size):
        la_sk = la_sk + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lask_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        la_cj = la_cj + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lacj_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        lm_phi = lm_phi + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmphi_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)
        lm_tht = lm_tht + np.loadtxt('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmtht_id'+str(id)+'_bach'+str(ite)+'.txt', delimiter=',').mean(axis=1)

    la_sk = la_sk/((sim//bach_size)-1)
    la_cj = la_cj/((sim//bach_size)-1)
    lm_phi = lm_phi/((sim//bach_size)-1)
    lm_tht = lm_tht/((sim//bach_size)-1)


    la_sk = la_sk.reshape(2,k)
    la_cj = la_cj.reshape(j,1)
    lm_tht = lm_tht.reshape(j,k)
    lm_phi = lm_phi.reshape(v,k)


    '''Predictions on testing set'''
    lm_tht_pred = np.repeat(0.5,test.shape[0]*k).reshape(test.shape[0],k)
    test0 = np.matrix(test)

    for j in np.arange(test.shape[0]):
        # intialise data of lists.
        sim_list = list(cosine_similarity(test0[j,:], train0)[0])
        sim_list= pd.DataFrame({'sim':sim_list})
        sim_list = sim_list.sort_values(by=['sim'],  ascending=False)
        lm_tht_pred[j,:] = lm_tht[list(sim_list.index[0:6])].mean(axis=0)

    y01_t_p = PGM_pred(lm_tht_pred,la_sk,la_cj,y01_t)
    ac = confusion_matrix(y01_t, y01_t_p)
    acc_sample.append((ac[0,0]+ac[1,1])/ac.sum())
    f1_sample.append(f1_score(y01_t, y01_t_p))

    with open('pgm_id12_f1.txt', 'w') as f:
        for item in f1_sample:
            f.write("%s\n" % item)

    with open('pgm_id12_acc.txt', 'w') as f:
        for item in acc_sample:
            f.write("%s\n" % item)


print('acc: ',acc_sample)
print('f1 : ', f1_sample)


'''
#PREDICTIONS FAILED APPROACH
#1) Goal: predict the theta value for each patient
#2) Using similarity between patients (memory based approach)
#3) using matrix multiplication don't work because some of the predicted values are negative.


def theta_predictions(phi,tht,testset):
    A = np.dot(phi,np.linalg.inv(np.dot(np.transpose(phi),phi)))
    B = np.dot(testset,A)
    scaler= sk.preprocessing.MinMaxScaler(feature_range=(tht.min(),tht.max())).fit(B)
    return(scaler.transform(B))

lm_tht_pred = theta_predictions(lm_phi,lm_tht, test)
print(lm_tht.min(),lm_tht.mean(),lm_tht.max(),lm_tht.var())
print(lm_tht_pred.min(),lm_tht_pred.mean(),lm_tht_pred.max(),lm_tht_pred.var())
print('testing set - ')
ac = acc(lm_tht_pred,la_sk,la_cj,y01_t)
print('acc using matrix multiplication',(ac[0,0]+ac[1,1])/ac.sum())

'''

#PLOTS
import plotnine as p9

def plot_chain_sk(location,size,i,id):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 1)
    la_array = la_array.iloc[[i,30+i]]
    la_array = la_array.transpose().reset_index(drop=True)
    la_array = la_array.unstack().reset_index()
    la_array.columns = ['parameter','sim','value']
    la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = p9.ggplot(la_array, p9.aes(x='sim',y='value' , color = 'parameter'))
    fig = fig + p9.geom_line()+p9.scale_y_continuous(limits = (lim[0],lim[1]))
    return fig

plot_chain_sk('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lask_id',sim//bach_size, 15,id)


def plot_chain_cj(location,size,i):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 0)
    la_array = la_array.iloc[:,i].reset_index(drop=True)
    la_array = la_array.reset_index(drop=False)
    la_array = la_array.reset_index(drop=True)
    la_array.columns = ['sim','value']
    #la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = (
           p9.ggplot(la_array,p9.aes(x='sim',y='value'))+
           p9.geom_line()+p9.scale_y_continuous(limits = (lim[0],lim[1]))
    )
    return fig

plot_chain_cj('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lacj_id',sim//bach_size, 15)

def plot_chain_tht(location,size,i):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 1)
    la_array = la_array.iloc[i]
    la_array = la_array.transpose().reset_index(drop=True)
    la_array = la_array.reset_index(drop=False)
    la_array.columns = ['sim','value']
    #la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = (
           p9.ggplot(la_array,p9.aes(x='sim',y='value'))+
           p9.geom_line()+p9.scale_y_continuous(limits = (lim[0],lim[1]))
    )
    return fig

plot_chain_tht('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmtht_id',sim//bach_size, 15)


def plot_chain_phi(location,size,i):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 1)
    la_array = la_array.iloc[i]
    la_array = la_array.transpose().reset_index(drop=True)
    la_array = la_array.reset_index(drop=False)
    la_array.columns = ['sim','value']
    #la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = (
           ggplot(la_array,aes(x='sim',y='value'))+
           geom_line()+scale_y_continuous(limits = (lim[0],lim[1]))
    )
    return fig

plot_chain_phi('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\output_lmphi_id',sim//bach_size, 15)
