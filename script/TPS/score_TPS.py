#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import os
import glob
import pandas as pd
import cPickle as pickle
import numpy as np
import sys

sys.path.append("../../model/OrdNMF")
sys.path.append("../../model/dcPF")
sys.path.append("../../model/function")

import preprocess_data as prep
import rec_eval

from scipy.stats import nbinom

#%%
def OrdNMF_loglik(yt,l,model):
    theta = model.theta
    delta = model.delta
    
    logP0 = np.log(1-np.exp(-theta[0]*l))
    logP = -l*theta + np.log(1-np.exp(-delta*l))
    loglik = logP-logP0
    #lik = np.exp(loglik)
    res = loglik[yt]
    return res.sum()

def dcPF_loglik(yt,link_RQ,l,p):
    N,trash = link_RQ.shape
    nb_pmf = nbinom.pmf(np.arange(N),-l/np.log(1-p),1-p)
    logP = np.log(nb_pmf.dot(link_RQ))
    logP0 = np.log(1-np.exp(-l))
    loglik = logP-logP0
    res = loglik[yt]
    return res.sum()

#%% Pre-processed data 
with open('../../data/TPS/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']
U,I = Y.shape

threshold = np.array([[0,1,2,5,10,20,50,100,200,500]])
compare = Y.data[:,np.newaxis]>threshold
Y.data = np.sum(compare,1)
T = Y.max()

# For dcPF
N = 1000
preproc = np.append(np.append(-float('inf'),threshold),float('inf'))
c = preproc[np.newaxis,:]< np.arange(N)[:,np.newaxis]
link_RQ = (1-c[:,1:])*c[:,:-1]

#%%
prop_test = 0.2
Seed_train_test = [1001] # seed

for seed_train_test in Seed_train_test:
    Y_train,Y_test = prep.divide_train_test(Y,prop_test=prop_test,seed=seed_train_test)
    u,i = Y_test.nonzero()
    yt = (Y_test.data[:,np.newaxis] == np.arange(T+1)[np.newaxis,:])
    
    save_dir = 'out/seed_%d' %(seed_train_test)

    for filename in glob.glob(os.path.join(save_dir,'*')):  
        print filename
        save=False
        with open(filename,'rb') as f:
            model = pickle.load(f)
            W = model.Ew
            H = model.Eh
        #model.score={} - erase the score
        for s in range(10):
            if ~np.isin('ndcg@100s'+str(s), model.score.keys()):
                save=True
                ndcg = rec_eval.normalized_dcg_at_k(Y_train>0,Y_test>s,W,H,k=100)
                model.score['ndcg@100s'+str(s)]=ndcg
        if model.classname == 'ONMF_implicit' and model.T==T:
            if ~np.isin('loglik', model.score.keys()):
                save=True
                l = np.sum(W[u,:]*H[i,:],1,keepdims=True)
                loglik = OrdNMF_loglik(yt,l,model)
                model.score['loglik']=loglik
        if model.classname == 'dcpf_Log' and model.filename[:7] == 'TPS_raw':
            if ~np.isin('loglik', model.score.keys()):
                save=True
                l = np.sum(W[u,:]*H[i,:],1,keepdims=True)
                loglik = dcPF_loglik(yt,link_RQ,l,model.p)
                model.score['loglik']=loglik
        if save == True:
            model.save_dir = save_dir
            model.save_model()

#%% Read scores
appended_data =[]
for seed_train_test in Seed_train_test:
    save_dir = 'out/seed_%d' %(seed_train_test)
    for filename in glob.glob(os.path.join(save_dir,'*')):  
        with open(filename,'rb') as f:
            model = pickle.load(f)
        df_name = pd.DataFrame.from_dict([{'filename':filename, 'classname':model.classname}])
        df_init = pd.DataFrame.from_dict([model.saved_args_init])
        df_fit = pd.DataFrame.from_dict([model.saved_args_fit])
        df_score = pd.DataFrame.from_dict([model.score])
        df_loc = pd.concat([df_name,df_init,df_fit,df_score], axis=1)
        if model.classname =='dcpf_Log':
            df_loc['en moyen'] = model.En.data.mean()
            df_loc['T'] = 0
            df_loc['approx'] = False
        else:
            df_loc['en moyen'] = model.En.mean()
        df_loc['Niter'] = len(model.Elbo)
        appended_data.append(df_loc)
      
if appended_data!=[]:
    df = pd.concat(appended_data, axis=0)

#%%
columns = [u'classname', u'filename', u'prefix',
           u'K', u'T', u'approx', 
           u'en moyen', u'Niter',
           u'ndcg@100s0', u'ndcg@100s2',u'ndcg@100s3',
           u'ndcg@100s4', u'ndcg@100s5', u'ndcg@100s6',
           u'prec_at_100', u'loglik']
df2 = df[columns] 

res = df2.groupby(['classname','prefix','T','approx','K']).mean().reset_index()
res_std = df2.groupby(['classname','prefix','T','approx','K']).std().reset_index()
verif_count = df2.groupby(['classname','prefix','T','approx','K']).count().reset_index()

#%%
sc = 'ndcg@100s0'
idx = res.groupby(['classname','prefix','T','approx'])[sc].transform(max) == res[sc]
res = res[idx]
res_std = res_std[idx]
