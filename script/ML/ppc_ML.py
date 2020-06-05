#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import os
import glob
import pandas as pd
import cPickle as pickle
import scipy.sparse as sparse
import numpy as np
import sys

sys.path.append("../../model/OrdNMF")
sys.path.append("../../model/dcPF")
sys.path.append("../../model/function")

import preprocess_data  as prep

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

import matplotlib
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

#%%
prop_test = 0.2
seed_test = 1001

with open('../../data/ML/ml_145_U2.00e+04_I1.19e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']
U,I = Y.shape
T = Y.max()

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
#%% OrdNMF
filename = 'ML_ONMF_implicit_K150_T10_alpha0.30_0.30_beta1.00_1.00_opthyper_beta_approxN_False_tol1.0e-05_seed4'
with open(os.path.join(save_dir,filename),'rb') as f:
    model = pickle.load(f)
    
W = model.Ew
H = model.Eh
L = W.dot(H.T)

theta = model.theta

#%%
np.random.seed(0)
Lstar = L/np.random.gamma(1,1,(U,I))

Yppc = np.zeros((U,I))
for t in theta:
    print t
    Yppc = Yppc + (Lstar>1./t)

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
ax.set_ylim([10**4, 10**7])

index = np.arange(n_groups)
bar_width = 0.3
opacity = .5

rects1 = ax.bar(index_truth, h_truth, bar_width,
                bottom=1,
                alpha=opacity,color='b',
                label=r"Truth ($%.2f$)"%sparsity_truth)
rects2 = ax.bar(index_onmf + bar_width, h_ppc_onmf, bar_width,
                bottom=1,
                alpha=opacity,color='r',
                label=r"OrdNMF ($%.2f$)"%sparsity_onmf)

plt.xlabel('Class')
plt.ylabel("Occurence number")
plt.xticks(index+1.1 ,np.arange(n_groups)+1)
plt.legend(loc=1)

plt.tight_layout()

plt.savefig('fig/PPC_ML.pdf',format='pdf', dpi=1200)
