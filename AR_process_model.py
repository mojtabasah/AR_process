# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:02:39 2020

@author: msahr
"""

import numpy as np

class AR_process_base():
    def __init__(self, N, p=1, sparsity=0.1, mat_list='s_sparse_Gaussian', 
                 fun='logistic', use_bias=False, D=None, **kwargs):
        self.N = N
        self.p = p
        self.D = D
        if D is None:
            self.use_D = False
            self.L = None
            self.mat_size = self.p
        else:
            self.use_D = True
            self.L = D.shape[1]
            self.mat_size = self.L
        self.sparsity = sparsity
        self.use_bias = use_bias
        self.mat, self.bias = self.create_mat(mat_list, **kwargs)
        self.fun = self.create_inv_link(fun)
        self.sample = None
        
    def create_mat(self, mat_list, **kwargs):
        if isinstance(mat_list, list):   #if a list of matrices is provided, just use it
            if self.use_bias:
                mat = np.array([mat[0] for mat in mat_list])
                bias = np.array([mat[1] for mat in mat_list])
            else:
                mat = np.array(mat_list)
                bias = np.zeros((self.N,))
        else:
            if mat_list == 'Bernoulli_Gaussian':
                mat = 2*np.random.normal(size=(self.N, self.N, self.mat_size),**kwargs)
                sparsity_pattern = np.random.binomial(1, self.sparsity, size=(self.N, self.N, self.mat_size))
                mat = mat*sparsity_pattern
            elif mat_list == 's_sparse_Gaussian':
                # Makes the matrix for each "neuron" exactly s-sparse
                s = int(self.sparsity*self.N*self.mat_size)
                mat = 2*np.random.normal(size=(self.N, self.N, self.mat_size),**kwargs)
                sparsity_pattern = np.zeros(shape=(self.N, self.N, self.mat_size))
                x = np.tile(range(self.N), self.mat_size)
                y = np.repeat(range(self.mat_size), self.N)
                for i in range(self.N):
                    idx = np.random.choice(range(self.N*self.mat_size), size=s, replace=False)
                    sparsity_pattern[i, x[idx], y[idx]] = 1
                mat = mat*sparsity_pattern
                mat_sum = np.sum(np.reshape(np.abs(mat), (self.N, -1)), axis=1)
                mat_sum = mat_sum[:, None, None]
                mat = mat/mat_sum*20
            if self.use_bias:
                bias = np.random.normal(size=(self.N,))
            else:
                bias = np.zeros((self.N,))
        return mat, bias
    
    def create_inv_link(self, fun):
        if fun == 'logistic':
            fun = lambda x: 1/(1 + np.exp(-x))
        elif fun == 'identity':
            fun = lambda x: x
        elif fun == 'exp':
            fun = lambda x:np.exp(x)
        else:
            fun = fun
        return fun
    
    def simulate(self, T, initialization='random', burnin=100):
        X = np.zeros((self.N, T + self.p + burnin))
        if initialization=='random':
            prob = 0.1
            X[:, :self.p] = np.random.binomial(1, prob, size=(self.N, self.p))
        
        for t in range(burnin + T):
            X_lag = X[:, t:t+self.p]
            if self.use_D:
                X_lag = X_lag.dot(self.D)
            z = np.array([self.fun(np.sum(mat*X_lag) + bias) for mat, bias in zip(self.mat, self.bias)])
            X[:, self.p + t] = self.sample(z)
        
        self.X = X[:, self.p + burnin:]
        
class BAR_process(AR_process_base):
    def __init__(self, N, p=1, sparsity=0.1, mat_list='s_sparse_Gaussian', 
                 fun='logistic', use_bias=False, D=None, **kwargs):
        AR_process_base.__init__(self, N, p, sparsity, mat_list, 
                 fun, use_bias, D, **kwargs)
        self.sample = lambda z: np.random.uniform(size=z.shape) < (z)


class GAR_process(AR_process_base):
    def __init__(self, N, p=1, var=1, sparsity=0.1, mat_list='s_sparse_Gaussian', 
                 fun='logistic', use_bias=False, D=None, **kwargs):
        AR_process_base.__init__(self, N, p, sparsity, mat_list, 
                 fun, use_bias, D, **kwargs)
        self.var = var
        self.sample = lambda z: z + np.random.normal(scale=np.sqrt(var), size=z.shape)
          
class LAR_process(AR_process_base):
    def __init__(self, N, p=1, lam=1, sparsity=0.1, mat_list='s_sparse_Gaussian', 
                 fun='logistic', use_bias=False, D=None, **kwargs):
        AR_process_base.__init__(self, N, p, sparsity, mat_list, 
                 fun, use_bias, D, **kwargs)
        self.lam = lam
        self.sample = lambda z: z + np.random.laplace(scale=lam, size=z.shape)
        
class PAR_process(AR_process_base):
    def __init__(self, N, p=1, sparsity=0.1, clamp=100, mat_list='s_sparse_Gaussian', 
                 fun='exp', use_bias=False, D=None, **kwargs):
        AR_process_base.__init__(self, N, p, sparsity, mat_list, 
                 fun, use_bias, D, **kwargs)
        self.clamp = clamp
        self.sample = lambda z: np.random.poisson(z.clip(min=0, max=clamp))