#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert

Variational Inference algorithm for Discrete Compound Poisson Factorization (DCPF)

- DCPF MODEL:
W ~ Gamma(aphaW,betaW)      ## UxK (size of W)
H ~ Gamma(aphaH,betaH)      ## IxK
C ~ Poisson(t*W*H)          ## UxIxK 
N = sum(C)                  ## UxI
Y ~ ED(\theta, n*\kappa)    ## UxI

- VARIATIONAL INFERENCE:
p(W,H,C,N) \approx q(C|N)q(N)q(W)q(H)
where:
    q(W) = Gamma()
    q(H) = Gamma()
    q(C|N) = Mult()
    q(N) = ? depends on the choice of the EDM, 
        see: dcpf_Log, dcpf_ZTP, dcpf_Geo, dcpf_sNB
"""

import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import os
import time
import cPickle as pickle
import sys

class dcpf():
    def __init__(self, K, 
                 t=1.,
                 alphaW=1., alphaH=1., betaW=1., betaH=1.):
        """
        K (int) - number of latent factors
        t (float, >0) - hyperparameter 
        alphaW (float, >0) - shape parameter of the prior of W
        alphaH (float, >0) - shape parameter of the prior of H
        betaW (float, >0) - rate parameter of the prior of W
        betaH (float, >0) - rate parameter of the prior of H
        """
        self.K = K
        self.t = t
        self.alphaW = alphaW
        self.alphaH = alphaH
        self.betaW = betaW
        self.betaH = betaH
        self.score={}
        # Save arguments
        saved_args_init = locals()
        saved_args_init.pop('self')
        self.saved_args_init = saved_args_init
        
    def fit(self,Y,
            seed=None, 
            opt_hyper=[], 
            precision=10**(-5), max_iter=10**5, min_iter=0,
            verbose=False,
            save=True, save_dir='', prefix=None, suffix=None):
        """
        ------- INPUT VARIABLES -------
        Y (sparse matrix of size UxI) - Observed data
        
        ------- OPTIONAL VARIABLES -------
        seed (int)
        opt_hyper (list of float)
            't' - update of the hyper-parameter t
            'beta' - update of the scale parameters of the gamma prior of W and H
                    betaW of size U, betaH of size I
            'p' - update of the parameters of the element distribution (EDM)
        precision (float) - stopping criterion on the ELBO
        max_iter (int) - maximum iteration number
        min_iter (int) - minimum iteration number 
        save (bool) - Saving the final class object
        save_dir (str) - Path of the saved file
        prefix, suffix (str) - prefix and suffix to use in the name of the saved file
        
        ------- SAVED VARIABLES -------
        Ew, Elogw : Expectations: Ew = E[W] and Elogw = E[log(W)]
        Eh, Elogh : idem for variable H
        En : Expectation of the number of listening sessions En = E[N]
        Elbo : Evolution of the ELBO
        """
        self.seed = seed
        np.random.seed(seed)
        self.opt_hyper = opt_hyper
        self.verbose = verbose
        self.precision = precision
        self.save = save
        self.save_dir = save_dir
        self.filename = self.create_filename(prefix, suffix)
        # Save arguments
        saved_args_fit = locals()
        saved_args_fit.pop('self')
        saved_args_fit.pop('Y')
        self.saved_args_fit = saved_args_fit
        # Timer
        start_time = time.time()
        
        self.stirling_matrix(Y)
        
        # INITIALIZATION 
        U,I = Y.shape
        s_y = Y.sum()
        Ew = np.random.gamma(1.,1.,(U,self.K))
        Eh = np.random.gamma(1.,1.,(I,self.K))
        s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
        Elogw = np.log(Ew)
        Elogh = np.log(Eh)
        # Local
        en, Ec_sum_i, Ec_sum_u, elboN = self.q_N_Mult(Y,np.exp(Elogw),np.exp(Elogh))
        s_en = en.sum()
        # Elbo
        self.Elbo = [-float("inf")]
        
        # ITERATIONS
        for n in range(max_iter):
            # Timer
            if verbose:
                print('ITERATION #%d' % n)
                start_t = _writeline_and_time('\tUpdates...')
            # HYPER PARAMETERS
            if np.isin('t',opt_hyper):
                self.t = s_en / float(s_wh)
            if np.isin('beta',opt_hyper):
                self.betaW = self.alphaW/Ew.mean(axis=1,keepdims=True)
                self.betaH = self.alphaH/Eh.mean(axis=1,keepdims=True)
            if np.isin('p',opt_hyper):
                self.opt_param_xl(s_en, s_y)
            # GLOBAL VARIATIONAL PARAMETERS 
            Ew, Elogw, elboW = q_Gamma(self.alphaW , Ec_sum_i, 
                                       self.betaW, self.t*np.sum(Eh,axis=0))
            Eh, Elogh, elboH = q_Gamma(self.alphaH, Ec_sum_u,
                                       self.betaH, self.t*np.sum(Ew,axis=0))
            s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
            # LOCAL 
            en, Ec_sum_i, Ec_sum_u, elboN = self.q_N_Mult(Y,np.exp(Elogw),np.exp(Elogh))
            s_en = en.sum()
            # ELBO
            elbo = elboN - self.t*s_wh + elboW + elboH
            self.rate = (elbo-self.Elbo[-1])/np.abs(self.Elbo[-1])
            if verbose:
                print('\r\tUpdates: time=%.2f'% (time.time() - start_t))
                print('\tRate:' + str(self.rate))
            if elbo<self.Elbo[-1]:
                self.Elbo.append(elbo) 
                raise ValueError('ELBO DECREASING!')
            if np.isnan(elbo):
                raise ValueError('ELBO = NAN')
            elif self.rate<precision and n>=min_iter:
                self.Elbo.append(elbo) 
                break
            self.Elbo.append(elbo) 
        
        # OUTPUT
        u,i = Y.nonzero()
        self.En = sparse.csr_matrix((en,(u,i)),shape=Y.shape)
        self.Ew = Ew.copy()
        self.Eh = Eh.copy()
        self.Elogw = Elogw.copy()
        self.Elogh = Elogh.copy()
        self.duration = time.time()-start_time
        # Save
        if self.save:
            self.save_model()
           
    def stirling_matrix(self,Y):
        pass
     
    def q_N_Mult(self,Y,W,H):
        """ 
        q(C,N) = q(N)q(C|N)
        q(C|N) = Multinomial
        q(N) depends on the choice of the element distribution
        
        OUTPUT:
        en - data of the sparse matrix En
        Ec_sum_i = \sum_i E[c_{uik}]
        Ec_sum_u = \sum_u E[c_{uik}]
        """
        # Product
        u,i = Y.nonzero()
        s = self.t*np.sum(W[u,:]*H[i,:],1)
        # N ?
        en, elbo = self.c_en(Y,s)
        # Mult
        R = sparse.csr_matrix((en/s,(u,i)),shape=Y.shape) # UxI
        Ec_sum_u = self.t*((R.T).dot(W))*H 
        Ec_sum_i = self.t*(R.dot(H))*W 
        return en, Ec_sum_i, Ec_sum_u, elbo 

    def create_filename(self,prefix,suffix):
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
                '_p%.3e' % (self.p) + \
                '_alpha%.2f_%.2f' %(self.alphaW, self.alphaH) + \
                '_beta%.2f_%.2f' % (self.betaW, self.betaH) + \
                '_opthyper_' + '_'.join(sorted(self.opt_hyper)) + \
                '_precision%.1e' %(self.precision) + \
                '_seed' + str(self.seed) + suffix
            
    def save_model(self):
        with open(os.path.join(self.save_dir, self.filename), 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    # ABSTRACT FUNCTIONS
    def c_en(self,Y,s):
        pass

    def opt_param_xl(self, s_en, s_y):
        pass
    
    def generate(self):
        pass
    
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
 
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()
