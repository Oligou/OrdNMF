#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import os
import cPickle as pickle
import scipy.sparse as sparse
import numpy as np
import sys

sys.path.append("../../model/OrdNMF")
sys.path.append("../../model/dcPF")
sys.path.append("../../function")

from OrdNMF import Ord_generate

import preprocess_data  as prep

import matplotlib.pyplot as plt

#%%
prop_test = 0.2
seed_test = 1001

with open('../../data/TPS/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']
U,I = Y.shape

# Thresholding the data
threshold = np.array([[0,1,2,5,10,20,50,100,200,500]])
T = threshold.shape[1]
compare = Y.data[:,np.newaxis]>threshold
Y.data = np.sum(compare,1)

Y_train,Y_test = prep.divide_train_test(Y,prop_test=prop_test,seed=seed_test)
save_dir = 'out/seed_%d' %(seed_test)
y = Y_train.data
D = float(U*I)

####################
info = np.unique(y,return_counts=True)
sparsity_truth = Y_train.nnz/D*100.
index_truth = info[0]
h_truth = info[1]

####################
#%% Log
filename = 'TPS_raw_dcpf_Log_K150_p5.000e-01_alpha0.30_0.30_beta1.00_1.00_opthyper_beta_p_precision1.0e-05_seed0'
with open(os.path.join(save_dir,filename),'rb') as f:
    model = pickle.load(f)
    
#%%
W = model.Ew
H = model.Eh
p = model.p

#%% PPC
np.random.seed(0)
Ya = np.random.negative_binomial(-np.dot(W,H.T)/np.log(1-p),1-p)
Y = sparse.csr_matrix(Ya)

#%% Threshold
compare = Y.data[:,np.newaxis]>threshold
Y.data = np.sum(compare,1)

#%%
info = np.unique(Y.data,return_counts=True)
index_dcpf = info[0]
h_ppc_dcpf = info[1]
sparsity_dcpf = Y.nnz/D*100.

####################
#%% OrdNMF
filename = 'TPS_OrdNMF_K250_T10_alpha0.30_0.30_beta1.00_1.00_opthyper_beta_approxN_False_tol1.0e-05_seed0'
with open(os.path.join(save_dir,filename),'rb') as f:
    model = pickle.load(f)
    
W = model.Ew
H = model.Eh
L = W.dot(H.T)
theta = model.theta

#%%
np.random.seed(0)

Yppc = Ord_generate(L,theta[:-1])
Yppc = sparse.csr_matrix(Yppc)
yppc = Yppc.data

#%%
info = np.unique(yppc,return_counts=True)
index_onmf = info[0]
h_ppc_onmf = info[1]
sparsity_onmf = Yppc.nnz/D*100.

####################
#%% PLOT 
n_groups = 10

# create plot
fig, ax = plt.subplots()
ax.set_yscale('log')

index = np.arange(n_groups)
bar_width = 0.2
opacity = .5

rects1 = ax.bar(index_truth, h_truth, bar_width,
                bottom=1,
                alpha=opacity,color='b',
                label=r"Truth ($%.2f$)"%sparsity_truth)
rects3 = ax.bar(index_onmf + bar_width, h_ppc_onmf, bar_width,
                bottom=1,
                alpha=opacity,color='r',
                label=r"OrdNMF ($%.2f$)"%sparsity_onmf)
rects2 = ax.bar(index_dcpf + 2*bar_width, h_ppc_dcpf, bar_width,
                bottom=1,
                alpha=opacity,color='orange',
                label=r"dcPF ($%.2f$)"%sparsity_dcpf)

plt.xlabel('Class')
plt.ylabel("Occurence number")
plt.xticks(index+1.2 ,np.arange(n_groups)+1)
plt.legend(loc=1)

plt.tight_layout()

plt.savefig('fig/PPC_TPS.pdf',format='pdf', dpi=1200)

