#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert

x_l \sim Log(p) (logarithmic distribution)
which implies: y \sim sumLog(n,p)
"""

import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import dcpf

class dcpf_Log(dcpf.dcpf):
    
    def __init__(self, K, p, t=1.,
                 alphaW=1., alphaH=1., betaW=1., betaH=1.):
        """
        p (float) - p=exp(\theta) where \theta is the natural parameter of the EDM
        """
        assert p>=0 and p<=1
        self.p = p
        dcpf.dcpf.__init__(self,K=K, t=t,
                 alphaW = alphaW, alphaH = alphaH, betaW = betaW, betaH = betaW)
        self.classname = 'dcpf_Log'
        
    def c_en(self,Y,s):
        y = Y.data
        # Limit cases
        if self.p==0: # PF on raw data
            en = Y.data
            elbo = - np.sum(special.gammaln(y+1)) + np.sum(y*np.log(s))
        elif self.p==1:  # PF on binary data
            en = np.ones_like(y, dtype=np.float)
            elbo = np.sum(np.log(s))
        else: # 0 < p < 1 - Trade-off
            r = s/(-np.log(1-self.p))
            en = np.ones_like(y, dtype=np.float)
            en[y>1] = r[y>1]*(special.digamma(y[y>1]+r[y>1])-special.digamma(r[y>1]))  
            # ELBO
            elbo_cst = -np.sum(special.gammaln(y+1)) + Y.sum()*np.log(self.p)
            elbo = elbo_cst + np.sum(special.gammaln(y+r) - special.gammaln(r))
        return en, elbo
    
    def opt_param_xl(self,s_en,s_y):
        """" Hyper-parameter optimization : Newton algortithm """
        ratio = float(s_en)/s_y
        p = self.p
        cost_init = s_y*np.log(p) - s_en*np.log(-np.log(1.-p))
        for n in range(10):
            f = (1.-p)/p*(-np.log(1-p))
            grad = np.log(1-p)/(p**2) + 1/p
            delta = (f-ratio)/grad
            while p - delta < 0 or p - delta > 1:
                delta = delta/2
            p = p - delta
        cost = s_y*np.log(p) - s_en*np.log(-np.log(1.-p))
        # Is the p better?
        if cost>cost_init:
            self.p = p
        
    def generate(self):
        pc = np.random.negative_binomial(
                -np.dot(self.Ew,self.Eh.T)/np.log(1.-self.p), 
                self.p)
        return sparse.csr_matrix(pc)

#%% Synthetic example
if False:
    import matplotlib.pyplot as plt

    U = 1000
    I = 1000
    K = 3
    
    np.random.seed(93)
    W = np.random.gamma(1.,.1, (U,K))
    H = np.random.gamma(1.,.1, (I,K))
    L = np.dot(W,H.T)
    Ya = np.random.poisson(L)
    Y = sparse.csr_matrix(Ya)
        
    #%%
    model = dcpf_Log(K=K,p=0.2)
    model.fit(Y,verbose=True, opt_hyper=['p','beta'], save=False)
    
    #%%
    Ew = model.Ew
    Eh = model.Eh            
    Yr = np.dot(Ew,Eh.T)
    
    #%%
    plt.figure('Obs')
    plt.imshow(Ya,interpolation='nearest')
    plt.colorbar()
    plt.figure('Truth')
    plt.imshow(L,interpolation='nearest')
    plt.colorbar()
    plt.figure('Reconstruction')
    plt.imshow(Yr,interpolation='nearest')
    plt.colorbar()

    #%%
    plt.figure('elbo')
    plt.plot(model.Elbo)
    