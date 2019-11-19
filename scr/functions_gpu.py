#Code adapted from Deconfounder Tutorial

#import tensorflow as tf
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import statsmodels.api as sm

import tensorflow.compat.v2 as tf
from tensorflow_probability import edward2 as ed
from sklearn.datasets import load_breast_cancer
from pandas.plotting import scatter_matrix
from scipy import sparse, stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import sys
import os
#path = 'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019'
#sys.path.append(path+'\\scr')
#os.chdir(path)


import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 10,
                            'xtick.labelsize' : 6,
                            'ytick.labelsize' : 6,
                            'axes.titlesize' : 10})
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue",
               "amber",
               "crimson",
               "faded green",
               "dusty purple",
               "greyish"]
colors = sns.xkcd_palette(color_names)
sns.set(style="white", palette=sns.xkcd_palette(color_names), color_codes = False)

# set random seed so everyone gets the same number
import random
randseed = 123
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
tf.random.set_seed(randseed)
#tf.set_random_seed(randseed)

#data = pd.read_csv("data//tcga_train_gexpression_cgc_7k.txt", sep = ';')
data = load_breast_cancer()
num_fea = 10

df = pd.DataFrame(data["data"][:,:num_fea], columns=data["feature_names"][:num_fea])

#df = data.drop('abr', axis = 1)
df.shape

#dfy = data["y"]
dfy = data["target"]
dfy.shape, dfy[:100] # binary outcomes

# perimeter and area are highly correlated with radius
fea_cols = df.columns[[(not df.columns[i].endswith("perimeter")) \
                     and (not df.columns[i].endswith("area")) \
                     for i in range(df.shape[1])]]

dfX = pd.DataFrame(df[fea_cols])

#dfX = df.drop(['patients','y'], axis = 1)

print(dfX.shape, dfy.shape)
# standardize the data for PPCA
#X = dfX
X = np.array((dfX - dfX.mean())/dfX.std())

# randomly holdout some entries of X
num_datapoints, data_dim = X.shape

holdout_portion = 0.2
n_holdout = int(holdout_portion * num_datapoints * data_dim)

holdout_row = np.random.randint(num_datapoints, size=n_holdout)
holdout_col = np.random.randint(data_dim, size=n_holdout)
holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), \
                            (holdout_row, holdout_col)), \
                            shape = X.shape)).toarray()

holdout_subjects = np.unique(holdout_row)

x_train = np.multiply(1-holdout_mask, X)
x_vad = np.multiply(holdout_mask, X)

def ppca_model(data_dim, latent_dim, num_datapoints, stddv_datapoints):
    w = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                scale=tf.ones([latent_dim, data_dim]),
                name="w")  # parameter
    z = ed.Normal(loc=tf.zeros([num_datapoints, latent_dim]),
                scale=tf.ones([num_datapoints, latent_dim]),
                name="z")  # local latent variable / substitute confounder
    x = ed.Normal(loc=tf.multiply(tf.matmul(z, w), 1-holdout_mask),
                scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                name="x")  # (modeled) data
    return x, (w, z)

log_joint = ed.make_log_joint_fn(ppca_model)

latent_dim = 2#0
stddv_datapoints = 0.1

model = ppca_model(data_dim=data_dim,
                   latent_dim=latent_dim,
                   num_datapoints=num_datapoints,
                   stddv_datapoints=stddv_datapoints)

def variational_model(qw_mean, qw_stddv, qz_mean, qz_stddv):
    qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
    qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
    return qw, qz


log_q = ed.make_log_joint_fn(variational_model)

def target(w, z):
    """Unnormalized target density as a function of the parameters."""
    return log_joint(data_dim=data_dim,
                   latent_dim=latent_dim,
                   num_datapoints=num_datapoints,
                   stddv_datapoints=stddv_datapoints,
                   w=w, z=z, x=x_train)

def target_q(qw, qz):
    return log_q(qw_mean=qw_mean, qw_stddv=qw_stddv,
               qz_mean=qz_mean, qz_stddv=qz_stddv,
               qw=qw, qz=qz)


qw_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
qz_mean = tf.Variable(np.ones([num_datapoints, latent_dim]), dtype=tf.float32)
qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([num_datapoints, latent_dim]), dtype=tf.float32))

qw, qz = variational_model(qw_mean=qw_mean, qw_stddv=qw_stddv,
                           qz_mean=qz_mean, qz_stddv=qz_stddv)

def elbo_func(qw,qz):
    energy = target(qw, qz)
    entropy = -target_q(qw, qz)
    return energy + entropy


#energy = target(qw, qz)
#entropy = -target_q(qw, qz)

#elbo = energy + entropy

#https://stackoverflow.com/questions/55318273/tensorflow-api-v2-train-has-no-attribute-adamoptimizer
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.05)
#optimizer = tf.optimizers.Adam(learning_rate = 0.05)
train = tf.optimizers.Adam(learning_rate = 0.05).minimize(-elbo_func(qw,qz), var_list=[qw_mean,qz_mean,qw_stddv,qz_stddv])
#https://stackoverflow.com/questions/58722591/typeerror-minimize-missing-1-required-positional-argument-var-list
#train = optimizer.minimize(-elbo, var_list=[qw_mean,qz_mean,qw_stddv,qz_stddv])
#https://github.com/tensorflow/probability/issues/524
#https://github.com/tensorflow/tensorflow/issues/28068
#f_without_any_args = functools.partial(f_batch_tensorflow, x=X, A=C, B=D)
#optimizer.minimize(f_without_any_args, X)
#https://github.com/tensorflow/tensorflow/issues/29944
#https://stackoverflow.com/questions/50064792/the-elbo-loss-of-variational-inference-implemented-in-tensorflow-probability-ba
#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
#here

#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/logistic_regression.py

init = tf.global_variables_initializer()

t = []

num_epochs = 500

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_epochs):
        sess.run(train)
        if i % 5 == 0:
            t.append(sess.run([elbo]))

        w_mean_inferred = sess.run(qw_mean)
        w_stddv_inferred = sess.run(qw_stddv)
        z_mean_inferred = sess.run(qz_mean)
        z_stddv_inferred = sess.run(qz_stddv)

print("Inferred axes:")
print(w_mean_inferred)
print("Standard Deviation:")
print(w_stddv_inferred)

plt.plot(range(1, num_epochs, 5), t)
plt.show()

def replace_latents(w, z):

    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
        """Replaces the priors with actual values to generate samples from."""
        name = rv_kwargs.pop("name")
        if name == "w":
            rv_kwargs["value"] = w
        elif name == "z":
            rv_kwargs["value"] = z
        return rv_constructor(*rv_args, **rv_kwargs)

    return interceptor

n_rep = 100 # number of replicated datasets we generate
holdout_gen = np.zeros((n_rep,*(x_train.shape)))

for i in range(n_rep):
    w_sample = npr.normal(w_mean_inferred, w_stddv_inferred)
    z_sample = npr.normal(z_mean_inferred, z_stddv_inferred)

    with ed.interception(replace_latents(w_sample, z_sample)):
        generate = ppca_model(
            data_dim=data_dim, latent_dim=latent_dim,
            num_datapoints=num_datapoints, stddv_datapoints=stddv_datapoints)

    with tf.Session() as sess:
        x_generated, _ = sess.run(generate)

    # look only at the heldout entries
    holdout_gen[i] = np.multiply(x_generated, holdout_mask)


n_eval = 100 # we draw samples from the inferred Z and W
obs_ll = []
rep_ll = []
for j in range(n_eval):
    w_sample = npr.normal(w_mean_inferred, w_stddv_inferred)
    z_sample = npr.normal(z_mean_inferred, z_stddv_inferred)

    holdoutmean_sample = np.multiply(z_sample.dot(w_sample), holdout_mask)
    obs_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                        stddv_datapoints).logpdf(x_vad), axis=1))

    rep_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                        stddv_datapoints).logpdf(holdout_gen),axis=2))

obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)

pvals = np.array([np.mean(rep_ll_per_zi[:,i] < obs_ll_per_zi[i]) for i in range(num_datapoints)])
holdout_subjects = np.unique(holdout_row)
overall_pval = np.mean(pvals[holdout_subjects])
print("Predictive check p-values", overall_pval)

subject_no = npr.choice(holdout_subjects)
sns.kdeplot(rep_ll_per_zi[:,subject_no]).set_title("Predictive check for subject "+str(subject_no))
plt.axvline(x=obs_ll_per_zi[subject_no], linestyle='--')


# approximate the (random variable) substitute confounders with their inferred mean.
Z_hat = z_mean_inferred
# augment the regressors to be both the assigned causes X and the substitute confounder Z
X_aug = np.column_stack([X, Z_hat])

# holdout some data from prediction later
X_train, X_test, y_train, y_test = train_test_split(X_aug, dfy, test_size=0.2, random_state=0)


dcfX_train = sm.add_constant(X_train)
dcflogit_model = sm.Logit(y_train, dcfX_train)
dcfresult = dcflogit_model.fit_regularized(maxiter=5000)
print(dcfresult.summary())

res = pd.DataFrame({"causal_mean": dcfresult.params[:data_dim+1], \
                  "causal_std": dcfresult.bse[:data_dim+1], \
                  "causal_025": dcfresult.conf_int()[:data_dim+1,0], \
                  "causal_975": dcfresult.conf_int()[:data_dim+1,1], \
                   "causal_pval": dcfresult.pvalues[:data_dim+1]})
res["causal_sig"] = (res["causal_pval"] < 0.05)
res = res.T
res.columns = np.concatenate([["intercept"], np.array(dfX.columns)])
res = res.T
print(res)


# make predictions with the causal model
dcfX_test = X_test
dcfy_predprob = dcfresult.predict(sm.add_constant(dcfX_test))
dcfy_pred = (dcfy_predprob > 0.5)
print(classification_report(y_test, dcfy_pred))
