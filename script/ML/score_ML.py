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

from scipy.stats import poisson
import preprocess_data  as prep
import rec_eval

#%% Pre-processed data 
with open('../../data/ML/ml_145_U2.00e+04_I1.19e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']
U,I = Y.shape
T = Y.max()

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
        df_name = pd.DataFrame.from_dict([{'filename':filename, 
                                           'classname':model.classname}])
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
        appended_data.append(df_loc)
      
if appended_data!=[]:
    df = pd.concat(appended_data, axis=0)

#%%
columns = [u'classname', u'filename', u'prefix',
           u'K', u'T', u'approx', 
           u'en moyen',
           u'ndcg@100s0', u'ndcg@100s3',u'ndcg@100s5',
           u'ndcg@100s7', u'ndcg@100s9',
           u'loglik']
df = df[columns] 

res = df.groupby(['classname','prefix','T','approx','K']).mean().reset_index()

#%%
sc = 'ndcg@100s7'
idx = res.groupby(['classname','prefix','T','approx'])[sc].transform(max) == res[sc]
res = res[idx]
