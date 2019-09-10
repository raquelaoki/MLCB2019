#Example
import  numpy as np
cimport numpy as np
import cython


cdef gibbs (np.ndarray[np.float64_t, ndim=2] c_la_sk, np.ndarray[np.float64_t, ndim=1] c_la_cj, np.ndarray[np.float64_t, ndim=2] c_lm_tht,
           np.ndarray[np.float64_t, ndim=2] c_lm_phi,  np.ndarray[np.float64_t, ndim=2] train0,
           int j,int v, int k, np.ndarray[np.int_t, ndim=1] y01):
    cdef np.ndarray[np.float64_t, ndim=2,negative_indices=False] n_la_sk = c_la_sk
    cdef np.ndarray[np.float64_t, ndim=1,negative_indices=False] n_la_cj = c_la_cj
    cdef np.ndarray[np.float64_t, ndim=2,negative_indices=False] n_lm_tht = c_lm_tht
    cdef np.ndarray[np.float64_t, ndim=2,negative_indices=False] n_lm_phi = c_lm_phi
    
    cdef np.ndarray[np.float64_t, ndim=3,negative_indices=False] lvjk = np.zeros((v,j,k))
    
    for ki in np.arange(k):
        lvjk[:,:,ki] = np.dot(c_lm_phi[:,ki].reshape(v,1), c_lm_tht[:,ki].reshape(1,j))       
    #check sum of poisson. I might be able to apply poisson after the sum, so will be faster
    lvjk = np.random.poisson(lvjk)
    cdef np.ndarray[np.float64_t, ndim=2,negative_indices=False] lvk = lvjk.sum(axis=1)
    cdef np.ndarray[np.float64_t, ndim=2,negative_indices=False] ljk = lvjk.sum(axis=0)
    for ki in np.arange(k):    
        n_lm_phi[:,ki] = np.random.dirichlet(alpha = (lvk[:,ki]+1.0004),size = 1)
        n_lm_tht[:,ki] = np.random.gamma(shape=(c_la_sk[y01,ki]+ljk[:,ki]).reshape(j),
                  scale=(np.divide(c_la_cj,1-c_la_cj)).reshape(j))
    
    cdef np.float64_t a2 = 1000000 #12000 before and average of 33
    cdef np.float64_t b2 = 100000000
    #it shoud be +
    cdef np.float64_t b2u = (np.log(np.divide(c_la_cj ,c_la_cj+np.log(1-0.1)))).sum()
    cdef np.ndarray[np.float64_t, ndim=1,negative_indices=False] uk
    cdef np.float64_t p
    
    for ki in np.arange(k):       
        uk = np.array([0,0])
        for ji in np.arange(j):
            p = c_la_sk[y01[ji],ki]/(c_la_sk[y01[ji],ki]+np.arange(max(ljk[ji,ki],1))+1)
            uk[y01[ji]] = uk[y01[ji]]+np.random.binomial(1,p=p).sum()
        n_la_sk[:,ki] = 1/np.random.gamma(a2+uk,1/(b2-b2u))     
        
    cdef np.float64_t a1 = 4000
    cdef np.float64_t b1 = 10000
    n_la_cj = np.random.beta(a= (a1+train0.sum(axis = 1)).reshape(j,1) ,b=(b1+n_lm_tht.sum(axis =1)).reshape(j,1))
    return(n_la_sk,n_la_cj,n_lm_tht,n_lm_phi)