# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:01:43 2020

@author: msahr
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso


class BAR_regressor():
    def __init__(self, N, p, D=None, use_bias=True):
        self.p = p
        self.N = N
        if D is None:
            self.D = None
            self.use_D = False
            self.L = None
            self.mat_size = p
        else:
            self.D = D
            self.use_D = True
            self.L = D.shape[1]
            self.mat_size = self.L
        self.mat = None
        self.fit_intercept = use_bias
        self.bias = None
        self.step = None
        self.method = None
        
    def fit(self, X, step=1, lam=1, method='logistic', **kwargs):
        self.method = method
        self.step = step
        T = X.shape[1] - self.p -self.step + 1
        mat_list = []
        intercept_list = []
        if self.use_D:
            data_mat = [(X[:, t:t + self.p - step + 1].dot(self.D)).flatten() for t in range(T)]
        else:
            data_mat = [X[:, t:t + self.p - step + 1].flatten() for t in range(T)]
        if method == 'logistic':
            C = 1/(lam*np.sqrt(T))
            regressor = LogisticRegression(penalty='l1', tol=0.0001, C=C, 
                                           fit_intercept=self.fit_intercept, solver='saga')
            for i in range(self.N):
                regressor.fit(data_mat, X[i + step - 1, self.p:])
                mat = regressor.coef_
                mat = np.reshape(mat, (self.N, self.mat_size))
                mat_list.append(mat)
                if self.fit_intercept:
                    intercept = regressor.intercept_
                    intercept_list.append(intercept)
        elif method == 'least_squares':
            alpha = 2*lam/np.sqrt(T)
            regressor = Lasso(alpha=alpha, fit_intercept=self.fit_intercept)
            for i in range(self.N):
                regressor.fit(data_mat, X[i + step - 1, self.p:])
                mat = regressor.coef_
                mat = np.reshape(mat, (self.N, self.mat_size))
                mat_list.append(mat)
                if self.fit_intercept:
                    intercept = regressor.intercept_
                    intercept_list.append(intercept)
        elif method == 'pytorch':
            pass
        else:
            raise ValueError('method not implemented yet or not valid.')
            
        self.mat = mat_list
        self.bias = np.array(intercept_list)
                
    def SE(self, theta):
        """
        Args
        ----------
        theta : array or arraylike
            ground truth tensor generating the process.

        Returns
        -------
        SE: scalar
            squared error
        NSE: scalar
            normalized square error

        """
        if self.mat is None:
            print('Run fit first. Estimated theta not found.')
            return
        theta = np.array(theta)
        thetahat = np.array(self.mat)
        error = theta - thetahat
        SE = np.sum(error**2)
        NSE = SE/np.sum(theta**2)
        return SE, NSE
    
    def create_inv_link(self, fun):
        if fun == 'logistic':
            fun = lambda x: 1/(1 + np.exp(-x))
        elif fun == 'identity':
            fun = lambda x: x
        else:
            fun = fun
        return fun
    
    def predict(self, X, step=1):
        """
        Predicts the process "step" steps in the future using data X, i.e.
        it uses X[:, t:t+p] to predict X[:, t+p+step-1].
        Note that step is the the number of steps in the furture to be predicted,
        whereas self.step is the number of steps in the future the regressor
        is trained to predict.

        Args
        ----------
        X : numpy array
            The data with dimension self.N*T  where T is the length of the sequence.
        step : int, optional
            How many steps in the future to predict. The default is 1.

        Raises
        ------
        ValueError
            if the steps tp be predicted and the steps the regressor is trained
            to predict (self.step) are not compatible. This happens when both of
            the folowing are true:
                - self.step != 1;
                - self.step != step.

        Returns
        -------
        None.

        """
        if (step != self.step and self.step != 1):
            raise ValueError(f'The regressoer is fitted to predict {self.step} steps in the future.' + 
                             f' This function call needs {step} steps pediction. They should either' + 
                             f' be equal or the regressor fitted to predict 1 step in the future!')
        r = step if self.step == 1 else 1
        T = X.shape[1] -self.p + 1
        fun = self.create_inv_link(self.method)
        theta = np.array(self.mat).reshape(self.N, self.N*self.mat_size)
        data = [X[:, t:t + self.p] for t in range(T)]
        
        Xhat = np.array([np.c_[data[t], np.zeros((self.N, r))] for t in range(T)])
        for t in range(r):
            if self.use_D:
                datahat = (Xhat[:, :, t:t+self.p].dot(self.D)).reshape(-1, self.N*self.L)
            else:
                datahat = Xhat[:, :, t:t+self.p].reshape(-1, self.N*self.p)
            zhat = fun(datahat.dot(theta.T) + self.bias.T)
            Xhat[:, :, self.p + t] = np.random.uniform(size=(T, self.N)) < (zhat)
        Xhat = np.swapaxes(Xhat[:, :, self.p:], 0, 1)
        return Xhat
    
    def score(self, Xhat, X):
        """
        Computes the accuracy of prediction from true process X and estimated
        process Xhat. Read description of X and Xhat to make sure they are 
        properly aligned.

        Parameters
        ----------
        Xhat : 3D numpy array
            It is an array with dimensions N*T*s where T is the number of samples 
            predicted and s is how many steps in the future the process is predicted.
            s can be greater than 1 iff self.step == 1, i.e. if the regressor is 
            trained to predict 1 step in the future.
        X : 2D numpy array
            The ground truth process values. It has dimesnsions 
                - N*(T+s-1) if self.step == 1
                - N*T if self.step == s where s is how many steps in the future
                the regressor is trained to predict. 
            Note that X should be aligned with Xhat, in the sense that X[:, t]
            should correspond to X[:, t, 0], i.e. the time indices of X are aligned
            with time indices of the matrix X[:,: 0] even though the time lengths
            are different when the number of steps predicted is greater than 1.

        Raises
        ------
        ValueError
            When there are different kinds of dimension msimatches..

        Returns
        -------
        acc : List
            a list of numpy arrays of shape (N,), and length s, where s is the
            number of steps predicted in the future if self.step == 1 and 1 otherwise.

        """
        r = Xhat.shape[2]
        T = X.shape[1]
        if X.shape[0] != Xhat.shape[0]:
            raise ValueError(f'Xhat.shape[0] = {Xhat.shape[0]} does' + 
                             ' not match X.shape[0] = {X.shape[0]}.')
        if T != Xhat.shape[1] + r - 1:
            raise ValueError('time dimension of X does not match time dimension of Xhat.')
        if self.step != 1 and r != 1:
            raise ValueError('self.step and Xhat shape are not compatible.')
        elif self.step == 1:
            acc = [np.sum(X[:, t:T + t - r + 1] == Xhat[:, :, t], axis=1)/(T -r + 1) for t in range(r)]
        else:
            acc = np.sum(X == Xhat[:, :, 0], axis=1)/T
        return np.array(acc)
