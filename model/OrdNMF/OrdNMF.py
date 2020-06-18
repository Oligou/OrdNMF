#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert

Variational Inference algorithm for Ordinal Non-Negative Matrix Factorization (OrdNMF)

- DCPF MODEL:
W ~ Gamma(aphaW,betaW)      ## UxK (size of W)
H ~ Gamma(aphaH,betaH)      ## IxK
C ~ OrdNMF(W*H)             ## UxI

- VARIATIONAL INFERENCE:
p(W,H,C,N) \approx q(C|N)q(N)q(W)q(H)
where:
    q(W) = Gamma()
    q(H) = Gamma()
    q(C|N) = Mult()
    q(N) = ZTP()
"""

#%% Ordinal NMF

import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import os
import time
import cPickle as pickle
import sys

class OrdNMF():
    def __init__(self, K,
                 alphaW = 1., alphaH = 1., betaW=1., betaH = 1.):
        """
        K (int) - number of latent factors
        alphaW (float, >0) - shape parameter of the prior of W
        alphaH (float, >0) - shape parameter of the prior of H
        betaW (float, >0) - rate parameter of the prior of W
        betaH (float, >0) - rate parameter of the prior of H
        """
        self.K = K
        self.alphaW = alphaW
        self.alphaH = alphaH
        self.betaW = betaW
        self.betaH = betaH
        self.score={}
        self.classname = 'OrdNMF'
        # Save arg
        saved_args_init = locals()
        saved_args_init.pop('self')
        self.saved_args_init = saved_args_init
        
    def fit(self, Y, T, 
            seed=None, 
            opt_hyper = ['beta'],
            approx = False,
            precision=10**(-5), max_iter=10**5, min_iter=0,
            verbose=False,
            save=True, save_dir='', prefix=None, suffix=None):
        """
        ------- INPUT VARIABLES -------
        Y (sparse matrix of size UxI, type:int) - Observed data, values from 0 to T
        T - maximum value in Y
        
        ------- OPTIONAL VARIABLES -------
        seed (int)
        opt_hyper (list of float)
            'beta' - update the scale parameters of the gamma prior of W and H
                    betaW of size U, betaH of size I
            'betaH' - update the scale parameters of the gamma prior of H
                    betaH is a scalar
        approx (bool) - if True, the variable N is approximated by a dirac located in 1
        precision (float) - stopping criterion on the ELBO
        max_iter (int) - maximum iteration number
        min_iter (int) - minimum iteration number 
        save (bool) - Saving the final class object
        save_dir (str) - Path of the saved file
        prefix, suffix (str) - prefix and suffix to use in the name of the saved file
        
        ------- SAVED VARIABLES -------
        Ew, Elogw : Expectations: Ew = E[W] and Elogw = E[log(W)]
        Eh, Elogh : idem for variable H
        Elbo : Evolution of the ELBO
        """
        self.seed = seed
        np.random.seed(seed)
        self.T = T
        self.opt_hyper = opt_hyper
        self.approx = approx
        self.verbose = verbose
        self.precision = precision
        # Save
        self.save = save
        self.save_dir = save_dir
        self.prefix = prefix
        self.suffix = suffix
        self.filename = self.filename(prefix, suffix)
        # Save arg
        saved_args_fit = locals()
        saved_args_fit.pop('self')
        saved_args_fit.pop('Y')
        self.saved_args_fit = saved_args_fit
        # Timer
        start_time = time.time()
                
        # Shape
        U,I = Y.shape
        u,i = Y.nonzero()
        y = Y.data
        # Init - matrix companion
        delta = self.init_delta(Y)        #delta = np.ones(T+1); delta[0]=0;
        H = (np.triu(np.ones((T+1,T+1))).dot(delta[:,np.newaxis]))[:,0] 
        theta0 = H[0]
        G = theta0 - H
        Gy = sparse.csr_matrix((G[y],(u,i)), shape=(U,I))
        # Init - W & H
        Ew = np.random.gamma(1.,1.,(U,self.K))
        Eh = np.random.gamma(1.,1.,(I,self.K))
        s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
        
        # Local
        Sw, Sh, En, elboLoc = self.q_loc(Y,delta,Ew,Eh)
            
        self.Elbo = [-float("inf")]
        self.info = []
        for n in range(max_iter):
            # Time
            if verbose:
                print('ITERATION #%d' % n)
                start_t = _writeline_and_time('\tUpdates...')
            # Hyper parameter
            if np.isin('beta',opt_hyper):
                self.betaW = self.alphaW/Ew.mean(axis=1,keepdims=True)
                self.betaH = self.alphaH/Eh.mean(axis=1,keepdims=True)
            if np.isin('betaH',opt_hyper):
                self.betaH = self.alphaH / np.mean(Eh)
            # Updates Delta
            lbd = np.sum(Ew[u,:]*Eh[i,:],1)
            S_lbd = s_wh
            for l in range(T,0,-1): # {T,...,1}
                S_lbd = S_lbd - np.sum(lbd[Y.data==l+1])
                delta[l] = np.sum(En[Y.data==l])/S_lbd
            H = (np.triu(np.ones((T+1,T+1))).dot(delta[:,np.newaxis]))[:,0] 
            theta0 = H[0]
            G = theta0 - H
            Gy = sparse.csr_matrix((G[y],(u,i)), shape=(U,I))
            # Global updates
            Ew, Elogw, elboW = q_Gamma(self.alphaW , Sw, 
                                       self.betaW, theta0*np.sum(Eh,0,keepdims=True) - Gy.dot(Eh))
            Eh, Elogh, elboH = q_Gamma(self.alphaH, Sh,
                                       self.betaH, theta0*np.sum(Ew,0,keepdims=True) - Gy.T.dot(Ew))
            s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
            # Local updates
            Sw, Sh, En, elboLoc = self.q_loc(Y,delta,np.exp(Elogw),np.exp(Elogh))
            # Elbo update
            elbo = elboLoc - theta0*s_wh + np.sum(Ew*Gy.dot(Eh)) + elboW + elboH
            if n==0:
                self.rate = float('inf')
            else:
                self.rate = (elbo-self.Elbo[-1])/np.abs(self.Elbo[-1])
            if verbose:
                print('\r\tUpdates: time=%.2f'% (time.time() - start_t))
                print('\tRate:' + str(self.rate))
            if elbo<self.Elbo[-1]:
                self.Elbo.append(elbo) 
                raise ValueError('Elbo diminue!')
            if np.isnan(elbo):
                #pass
                raise ValueError('elbo NAN')
            elif self.rate<precision and n>=min_iter:
                self.Elbo.append(elbo) 
                break
            self.Elbo.append(elbo) 
            self.info.append(delta.copy())
        
        self.delta = delta
        self.theta = (np.triu(np.ones((T+1,T+1)),1).dot(delta[:,np.newaxis]))[:,0]
        self.Ew = Ew.copy()
        self.Eh = Eh.copy()
        self.En = En.copy()
        self.Elogw = Elogw.copy()
        self.Elogh = Elogh.copy()
        
        self.duration = time.time()-start_time
        
        # Save
        if self.save:
            self.save_model()
     
    def init_delta(self,Y):
        """ Initialization of delta w.r.t. the histogram values of Y  """
        hist_values = np.bincount(Y.data)
        hist_values[0] = Y.nnz
        cum_hist = np.cumsum(hist_values, dtype=float)
        delta = hist_values/cum_hist
        delta[0]=0
        return delta

    def q_loc(self,Y,delta,W,H):
        """ 
        q(C,N) = q(N)q(C|N)
        q(C|N) = Multinomial
        q(N) = ZTP
        
        OUTPUT:
        en - data of the sparse matrix En
        Sw = \sum_i E[c_{uik}]
        Sh = \sum_u E[c_{uik}]
        """
        # Product        
        u,i = Y.nonzero()
        Lbd = np.sum(W[u,:]*H[i,:],1)
        delta_y = delta[Y.data]
        # En
        if self.approx == False:
            en = Lbd*delta_y/(1.-np.exp(-Lbd*delta_y)) #delta_y/(1.-np.exp(-Lbd*delta_y))
            en[np.isnan(en)] = 1.
        else :
            en = np.ones_like(Lbd)
        # Sum C
        R = sparse.csr_matrix((en/Lbd,(u,i)),shape=Y.shape) # UxI
        Sw = W*(R.dot(H)) 
        Sh = H*(R.T.dot(W))
        # ELBO
        elbo = np.sum(np.log(np.expm1(Lbd*delta_y)))
        return Sw, Sh, en, elbo

    def filename(self,prefix,suffix):
        if prefix is not None:
            prefix = prefix+'_'
        else:
            prefix = ''
        if suffix is not None:
            suffix = '_'+suffix
        else:
            suffix = ''
        return prefix + self.classname + \
                '_K%d' % (self.K) + \
                '_T%d' % (self.T) + \
                '_alpha%.2f_%.2f' %(self.alphaW, self.alphaH) + \
                '_beta%.2f_%.2f' % (self.betaW, self.betaH) + \
                '_opthyper_' + '_'.join(sorted(self.opt_hyper)) + \
                '_approxN_' + str(self.approx) + \
                '_tol%.1e' %(self.precision) + \
                '_seed' + str(self.seed) + suffix
            
    def save_model(self):
        with open(os.path.join(self.save_dir, self.filename), 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def copy_attributes(self,oobj):
        self.__dict__ = oobj.__dict__.copy()
                
def stat_gamma(shape,rate):
    """
    Statistic of a gamma distribution:
        x \sim Gamma(shape, rate)
        INPUT: shape and rate parameters
        OUTPUT: E[x], E[log(x)], H the entropy
    """
    E = shape/rate
    dig_shape = special.digamma(shape)
    Elog = dig_shape - np.log(rate)
    entropy = shape - np.log(rate) + special.gammaln(shape) + (1-shape)*dig_shape
    return E, Elog, entropy
  
def gamma_elbo(shape, rate, Ex, Elogx):
    """ Part of the ELBO linked to the gamma prior """
    return (shape-1)*Elogx -rate*Ex +shape*np.log(rate) -special.gammaln(shape)

def q_Gamma(shape, _shape, rate, _rate):
    """ Calculate both statistic and ELBO """
    E,Elog,entropy = stat_gamma(shape+_shape, rate+_rate)
    elbo = gamma_elbo(shape, rate, E, Elog)
    elbo = elbo.sum() + entropy.sum()
    return E, Elog, elbo

def Ord_generate(L, theta):
    Y = np.zeros(L.shape)
    X = L/np.random.gamma(1,1,L.shape)
    for t in theta:
        Y = Y + (X>1./t)
    Y = Y.astype(int)
    return Y
    
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()
