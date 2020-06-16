#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

#%% 
import sys
sys.path.append("../../model/OrdNMF")
sys.path.append("../../function")

import os
import cPickle as pickle
import numpy as np
#import matplotlib.pyplot as plt

from OrdNMF import OrdNMF

import preprocess_data  as prep

#%% 
prop_test = 0.2
seed_test = 1001

with open('../../data/ML/ml_145_U2.00e+04_I1.19e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']
U,I = Y.shape

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

# Ordinal
if False:
    for approx in [False]: 
        for K in Ks:
            for seed in Seeds:
                model = OrdNMF(K=K, alphaW=alpha, alphaH=alpha)
                model.fit(Y_train, T=Y_train.max(), 
                          seed=seed, opt_hyper=opt_hyper, 
                          approx = approx, 
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                          save=True, save_dir=save_dir,prefix='ML', 
                          verbose=False)
   
# Binary  
if False:
    for approx in [True,False]: # Approx Bernoulli -> Poisson
        for K in Ks:
            for seed in Seeds:
                R1 = Y_train>=1
                R1.eliminate_zeros()
                model = OrdNMF(K=K, alphaW=alpha, alphaH=alpha)
                model.fit(R1, T=1, 
                          seed=seed, opt_hyper=opt_hyper, 
                          approx = approx, 
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                          save=True, save_dir=save_dir,prefix='ML_geq1', 
                          verbose=False)
        
if False:
    for approx in [True,False]: # Approx Bernoulli -> Poisson
        for K in Ks:
            for seed in Seeds:                
                R8 = Y_train>=8
                R8.eliminate_zeros()
                model = OrdNMF(K=K, alphaW=alpha, alphaH=alpha)
                model.fit(R8, T=1, 
                          seed=seed, opt_hyper=opt_hyper, 
                          approx = approx, 
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                          save=True, save_dir=save_dir,prefix='ML_geq8', 
                          verbose=False)
        