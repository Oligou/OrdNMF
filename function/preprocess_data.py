#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import numpy as np

def divide_train_test(Y,prop_test=0.2,seed=None):
    np.random.seed(seed)
    
    U,I = Y.shape
    nnz = Y.nnz
    
    mask = np.random.choice(nnz, size=int(nnz*prop_test), replace=False)
    no_mask = np.asarray(list(set(range(nnz)) - set(mask)))
    
    Y_train = Y.copy()
    Y_test = Y.copy()
    
    Y_train.data[mask]=0
    Y_train.eliminate_zeros()
    
    Y_test.data[no_mask]=0
    Y_test.eliminate_zeros()
    
    assert (Y - Y_train - Y_test).sum() == 0
    assert (Y_train.multiply(Y_test)).sum() == 0
    return Y_train,Y_test