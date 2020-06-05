#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

#%% 
import sys
sys.path.append("../../model/OrdNMF")
sys.path.append("../../model/dcPF")
sys.path.append("../../model/function")

import os
import cPickle as pickle
import numpy as np
#import matplotlib.pyplot as plt

from OrdNMF import OrdNMF
from dcpf_Log import dcpf_Log

import preprocess_data  as prep

#%% 
prop_test = 0.2
seed_test = 1001

with open('../../data/TPS/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']
U,I = Y.shape

# Thresholding the data
threshold = np.array([[0,1,2,5,10,20,50,100,200,500]])
compare = Y.data[:,np.newaxis]>threshold
Y.data = np.sum(compare,1)

#hist_values = np.bincount(Y.data)
#plt.figure()
#plt.semilogy(np.arange(1,Y.max()+1),hist_values[1:],'.-')

#%% Directory
Y_train,Y_test = prep.divide_train_test(Y,prop_test=prop_test,seed=seed_test)
save_dir = 'out/seed_%d' %(seed_test)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        
#%% IN 
opt_hyper = ['beta']  

Ks = [25,50,100,150,200,250]
Seeds = range(5) # Seed of the different initializations
tol = 10**(-5)
min_iter = 0
max_iter = 10**5
    
#%% Ord NMF
alpha = .3 

# OrdNMF 
if True:
    for approx in [False]: 
        for K in Ks:
            for seed in Seeds:
                model = OrdNMF(K=K, alphaW=alpha, alphaH=alpha)
                model.fit(Y_train, T=Y_train.max(), 
                          seed=seed, opt_hyper=opt_hyper, 
                          approx = approx, 
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                          save=True, save_dir=save_dir,prefix='TPS', 
                          verbose=False)
   
# PF or BePoF on binary  
if True:
    for approx in [True,False]: # Approx Bernoulli -> Poisson
        for K in Ks:
            for seed in Seeds:
                model = OrdNMF(K=K, alphaW=alpha, alphaH=alpha)
                model.fit(Y_train>0, T=1, 
                          seed=seed, opt_hyper=opt_hyper, 
                          approx = approx, 
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                          save=True, save_dir=save_dir,prefix='TPS_bin', 
                          verbose=False)
 
                    
#%% dcPF on raw
if True:
    with open('../../data/TPS/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
        out = pickle.load(f)
        Y = out['Y_listen']
        Y_train,Y_test = prep.divide_train_test(Y,prop_test=prop_test,seed=seed_test)
    
    for K in Ks:
        for seed in Seeds:
            if True:
                model = dcpf_Log(K=K, p=.5, alphaW=alpha,alphaH=alpha)
                model.fit(Y_train, 
                          seed=seed, opt_hyper = ['p','beta']  , 
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                          save=True, save_dir=save_dir,prefix='TPS_raw', 
                          verbose=False)
                    
