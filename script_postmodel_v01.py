'''Loading libraries'''
import pandas as pd 
import numpy as np 
#import time
import matplotlib.pyplot as plt
sys.path.append('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019')
#from script_v004_def import *

#important variables
k = 100
sim = 4000
bach_size = 1000
step1 = 10
step2 = 20
id = '0004'

filename = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final_log.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final.csv"
#filename = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data\\data_final_sub.csv"
data = pd.read_csv(filename, sep=',')


'''Splitting Dataset'''
#data = data.iloc[:, 0:1000]
#data = data.sample(n=1000).reset_index(drop=True)
data, test = train_test_split(data, test_size=0.3, random_state=42)
#print(data.shape, test.shape)


'''Organizing columns names'''
lr = data.columns[[2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y = data.columns[3]
remove = data.columns[[0,1]]


'''2 - Accuracy  Traning set'''
#Loading files names
files_p = []
files_tht = []
for ite in range(int(sim/bach_size)):
    files_p.append('Data\\output_p_id'+id+'_bach'+str(ite)+'.txt')
    files_tht.append('Data\\output_lmtht_id'+id+'_bach'+str(ite)+'.txt')
    
#Loading files
p_sim=pd.read_csv(files_p[0],sep=',', header=None)
tht_sim=pd.read_csv(files_tht[0],sep=',', header=None)
for i in range(1,len(files_p)):
    p_sim = pd.concat([p_sim,pd.read_csv(files_p[i],sep=',', header=None)],axis =0)
    tht_sim = pd.concat([tht_sim,pd.read_csv(files_tht[i],sep=',', header=None)],axis=1)

#phi: every column is a simulation, every row is a position in the matrix
tht_array = []
for i in range(20,tht_sim.shape[1]):
    tht_array.append(np.array(tht_sim.iloc[0:,i]).reshape(j,k))
theta = np.mean( tht_array , axis=0 )
p = p_sim.reset_index(drop=True).drop(range(20),axis=0).mean(axis=0)
#p = p_sim.iloc[0,:] 
   
col = ['intercept','gender', 'abr_ACC', 'abr_BLCA', 'abr_CHOL', 'abr_ESCA', 'abr_HNSC',
       'abr_LGG', 'abr_LIHC', 'abr_LUSC', 'abr_MESO', 'abr_PAAD', 'abr_PRAD',
       'abr_SARC', 'abr_SKCM', 'abr_TGCT', 'abr_UCS']
data['intercept']=np.repeat(1,data.shape[0])
data_P = pd.concat([data[col].reset_index(drop=True),pd.DataFrame(theta).reset_index(drop=True)],axis=1,ignore_index=True)


fit = 1/(1+np.exp(data_P.mul(p).sum(axis=1)))
#id = '0003'
#sim = 4000
#bach_size = 500


#checking seed 
p0 = p_sim.iloc[0,:] 
fit0 = 1/(1+np.exp(data_P.mul(p0).sum(axis=1)))
fit0.describe()



files = []
for ite in range(int(sim/bach_size)):
    files.append('Data\\output_lask_id'+id+'_bach'+str(ite)+'.txt')

f=pd.read_csv(files[0],sep=',', header=None)
for i in range(1,len(files)):
    f = pd.concat([f,pd.read_csv(files[i],sep=',', header=None)],axis =0,sort=False)


'''Plots'''
plt.plot(np.arange(0,f.shape[0]),f.iloc[:,6], 'r-', alpha=1)
plt.show()
plt.plot(np.arange(0,f.shape[0]),f.iloc[:,19], 'r-', alpha=1)
plt.show()
plt.plot(np.arange(0,f.shape[0]),f.iloc[:,96], 'r-', alpha=1)
plt.show()

#plt.plot(np.arange(0,len(output_logistic[:,0])),output_logistic[:,0], 'r-', alpha=1)
#plt.xlabel('Logistic Regression - intercept')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_intercept.png')

#plt.plot(np.arange(0,len(output_logistic[:,1])),output_logistic[:,1], 'r-', alpha=1)
##plt.xlabel('Logistic Regression - gender')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_gender.png')

#plt.plot(np.arange(0,len(output_logistic[:,10])),output_logistic[:,10], 'r-', alpha=1)
#plt.xlabel('Logistic Regression - cancer type (one of them)')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_cancertype.png')

#plt.plot(np.arange(0,len(output_logistic[:,30])),output_logistic[:,30], 'r-', alpha=1)
#plt.xlabel('Logistic Regression - k (one of them)')
#plt.show()
#plt.savefig('Data\\plot'+id+'lr_k.png')

#all_objects = muppy.get_objects()
#sum1 = summary.summarize(all_objects)
# Prints out a summary of the large objects
#summary.print_(sum1)
# Get references to certain types of objects such as dataframe
#dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
#for d in dataframes:
 # print(d.columns.values)
 # print(len(d))

#crash kernel
#all_objects = muppy.get_objects()
#sum1 = summary.summarize(all_objects)
#summary.print_(sum1)                          
 
 