# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:15:27 2020

@author: msahr
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
        
from AR_process_model import BAR_process, PAR_process
from BAR_scikit_regressor import BAR_regressor
        
class AR_torch_base(nn.Module):
    def __init__(self, N, p, D, device):
        super(AR_torch_base, self).__init__()
        self.N = N
        self.p = p
        if D is None:
            self.D = None
            self.use_D = False
            self.L = None
            self.mat_size = p
        else:
            self.D = torch.tensor(D, dtype=torch.float).to(device)
            self.use_D = True
            self.L = D.shape[1]
            self.mat_size = self.L
        self.device = device
        self.mat = None
        self.step = None
        self.method = None
        self.sample = None
        self.raw_score = None
        self.loss_hist = []
        self.reg_hist = []
        self.dtype = torch.float
        
    def loss(self, yhat, y, method):
        if method =='least_squares':
            return torch.sum( (yhat - y)**2 )
        elif method == 'logistic':
            return -torch.sum( y*torch.log(yhat) + (1 - y)*torch.log(1 - yhat) )
        elif method == 'Poisson':
            return torch.sum(yhat - y*torch.log(yhat))
        elif method == 'mae':
            return torch.sum(torch.abs((yhat - y)))
        else:
            raise ValueError(f'loss method = {method} not understood!')
            
    def create_optimizer(self, params, optimization_alg, lr, **kwargs):
        if optimization_alg == 'SGD':
            optimizer = optim.SGD(params, lr=lr, momentum=0, **kwargs)
        elif optimization_alg == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, momentum=0.2, **kwargs)
        elif optimization_alg == 'Adam':
            optimizer = optim.Adam(params, lr=lr, betas=(0.1, 0.99))
        else:
            raise ValueError('Optimization algorithm not recognized!')
        return optimizer
    
    def fit(self, X, step=1, lam=0.0002, epochs=50, batch_size=2000, method='least_squares', optimization_alg='SGD', lr=5e-2, 
              gamma=0.75, log_interval=100, proximal=False, hist=True, theta=None, **kwargs):
        self.method = method
        self.step = step
        T = torch.tensor([X.shape[1] - self.p - step + 1], dtype=torch.float)
        dataset = AR_dataset(X, self.N, self.p, D=self.D, step=self.step, 
                             dtype=torch.float, device=self.device)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                                                    
        optimizer = self.create_optimizer(self.parameters(), optimization_alg=optimization_alg, 
                                              lr=lr, **kwargs)
        C = (lam/torch.sqrt(T)).to(self.device)
        
        scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
        self.to(self.device)
        self.train()
        prev_loss = torch.tensor(np.inf).to(self.device)
        for epoch in range(1, epochs + 1):
            lr = torch.tensor(scheduler.get_lr()).to(self.device)
            prox = lambda x: torch.sign(x)*(torch.abs(x) - C*lr)*(torch.abs(x) > (C*lr))   #proximal operator of \ell_1 norm
            hard_thresh = lambda x: x*(torch.abs(x) > 0.001)
            if theta is not None:
                self.mat = list(self.fc.parameters())[0].detach().cpu().numpy().reshape((self.N, self.N, self.mat_size))
                SE, NSE = self.SE(theta)
                print(f'SE: {SE};    NSE: {NSE}')
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(data)
                regularization_loss = torch.tensor([0.0]).to(self.device)
                if not proximal:
                    for name, param in self.named_parameters():
                        if 'weight' in name:
                            regularization_loss += torch.sum(abs(param))
                loss = self.loss(output, target, method)/batch_size
                regularized_loss = loss + C*regularization_loss
                regularized_loss.backward()
                optimizer.step()
                if loss > 5*prev_loss:
                    adjust_lr(optimizer, loss, prev_loss)
                prev_loss = loss.detach()
                if hist:
                    self.loss_hist.append(loss.detach().cpu().numpy())
                    self.reg_hist.append(regularization_loss.detach().cpu().numpy())
                if proximal:
                    for param in self.parameters():
                        param.data = prox(param.data)
                        
                # for param in self.parameters():
                #         param.data = hard_thresh(param.data)
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  lr: {:.5f}'.format(
                       epoch, batch_idx * len(data), len(train_loader),
                        100. * batch_idx / len(train_loader), loss.item(), lr.item()))
            scheduler.step()
        
    def SE(self, theta):
        """
        Args
        ----------
        theta : array or arraylike
            ground truth tensor generating the process.

        Returns
        -------
        SE: scalar
            square error
        NSE: scalar
            normalized square error

        """
        if self.mat is None:
            print('Run fit first. Estimated theta not found.')
            return
        theta = np.array(theta)
        thetahat = self.mat
        error = theta - thetahat
        SE = np.sum(error**2)
        NSE = SE/np.sum(theta**2)
        return SE, NSE   
    
    def predict(self, X, step=1, runs=1):
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

        runs : int
            number of MCMC runs. The default is 1.
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
        Xhat : numpy array
            A tensor of size: runs*N*N_ts*steps where N is self.N, N_ts is the
            number of test samples computed from X.

        """
        Xhat_MCMC = [self._predict(X, step).detach().cpu().numpy() for _ in range(runs)]
        Xhat_MCMC = np.array(Xhat_MCMC)
        return Xhat_MCMC
    
    
    def _predict(self, X, step=1):
        self.eval()
        if (step != self.step and self.step != 1):
            raise ValueError(f'The regressoer is fitted to predict {self.step} steps in the future.' + 
                             f' This function call needs {step} steps pediction. They should either' + 
                             f' be equal or the regressor fitted to predict 1 step in the future!')
        r = step if self.step == 1 else 1
        T = X.shape[1] -self.p + 1
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        data = [X[:, t:t + self.p] for t in range(T)]
        Xhat = torch.stack([torch.cat([data[t], torch.zeros((self.N, r)).to(self.device)], dim=1) for t in range(T)])
        for t in range(r):
            datahat = Xhat[:, :, t:t+self.p]
            if self.use_D:
                datahat = torch.matmul(datahat, self.D)
            datahat = datahat.reshape(-1, self.N*self.mat_size)
            zhat = self(datahat)
            Xhat[:, :, self.p + t] = self.sample(zhat) #(torch.rand(T, self.N) < (zhat)) + 0.0
        Xhat = Xhat[:, :, self.p:].permute(1, 0, 2)
        return Xhat
    
    def score(self, Xhat, X, metric='accuracy'):
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
        s : List
            a list of numpy arrays of shape (N,), and length s, where s is the
            number of steps predicted in the future if self.step == 1 and 1 otherwise.

        """
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if not torch.is_tensor(Xhat):
            Xhat = torch.tensor(Xhat)
            
        if metric == 'accuracy':
            loss = lambda X, Xhat: (X == Xhat) + 0.0
        elif metric == 'mae':
            loss = lambda X, Xhat: torch.abs(X - Xhat) + 0.0
        elif metric == 'mse':
            loss = lambda X, Xhat: (X - Xhat)**2 + 0.0
        X, Xhat = X.type(self.dtype).to(self.device), Xhat.type(self.dtype).to(self.device)
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
            self.raw_score = torch.stack([loss(X[:, t:T + t - r + 1], Xhat[:, :, t]) for t in range(r)], dim=0)
        else:
            self.raw_score = loss(X, Xhat[:, :, 0])
        
        s, var = torch.mean(self.raw_score, axis=-1), torch.var(self.raw_score, axis=-1)
        return s.detach().cpu().numpy(), var.detach().cpu().numpy()
 
def adjust_lr(optimizer, loss, prev_loss):
    alpha = 1
    if loss > prev_loss*100:
        alpha = 0.1
    elif loss > prev_loss*10:
        alpha = 0.5
    if alpha != 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= alpha
        
               

class BAR_torch_regressor(AR_torch_base):
    def __init__(self, N, p, fun=None, use_bias=True, D=None, device='cpu'):
        """
        Bernoulli Autoregressive (BAR) process estimator. It estimates parameters 
        of a multivariate BAR.

        Parameters
        ----------
        N : int
            dimension of the Bernoulli process.
        p : int
            time lag. How many samples before sample t should be considered
            for estimating the mean paramters at time t.
        fun : function 
            The inverse link function of the process implemented in torch. The default is None.
        use_bias : boolean, optional
            Whether to use bias or not. The default is True.
        device : optional
            which device to save the model in. Use 'gpu{i}' if Cuda is available. 
            The default is 'cpu'.

        Returns
        -------
        None.

        """
        AR_torch_base.__init__(self, N, p, D, device)
        self.use_bias = use_bias
        if self.L is None:
            self.fc = nn.Linear(N*p, N, bias=use_bias)
        else:
            self.fc = nn.Linear(N*self.L, N, bias=use_bias)
        self.fun = fun
        self.bias = None
        self.sample = lambda x: ((torch.rand(x.shape).to(self.device) < x) + 0.0).to(device)
        
    def forward(self, x):
        x = self.fc(x)
        if self.fun is not None:
            x = self.fun(x)
        return x
    
                
    def fit(self, X, step=1, lam=0.0002, epochs=50, batch_size=2000, method='least_squares', optimization_alg='SGD', lr=5e-2, 
              gamma=0.8, log_interval=100, proximal=False,**kwargs):
        AR_torch_base.fit(self, X, step=step, lam=lam, epochs=epochs, batch_size=batch_size, 
                          method=method, optimization_alg=optimization_alg, lr=lr, 
                          gamma=gamma, log_interval=log_interval, proximal=proximal,**kwargs)

        self.mat = list(self.fc.parameters())[0].detach().cpu().numpy().reshape((self.N, self.N, self.mat_size))
        if self.use_bias:
            self.bias = list(self.fc.parameters())[1].detach().cpu().numpy()

class PAR_torch_regressor(AR_torch_base):
    def __init__(self, N, p, clamp=None, fun=None, use_bias=True, D=None, device='cpu'):
        """
        Poisson Autoregressive (BAR) process estimator. It estimates parameters 
        of a multivariate PAR.

        Parameters
        ----------
        N : int
            dimension of the Bernoulli process.
        p : int
            time lag. How many samples before sample t should be considered
            for estimating the mean paramters at time t.
        clamp: float
            Poisson process can easily become unstable if rates are not clamped.
            Use a value here to clamp the rates (currently a single clamp value
                                                 for all dimension is supported only)
            Default is None where no clamp is applied.
        fun : function 
            The inverse link function of the process implemented in torch. The default is None.
        use_bias : boolean, optional
            Whether to use bias or not. The default is True.
        device : optional
            which device to save the model in. Use 'gpu{i}' if Cuda is available. 
            The default is 'cpu'.

        Returns
        -------
        None.

        """
        AR_torch_base.__init__(self, N, p, D, device)
        self.use_bias = use_bias
        if self.L is None:
            self.fc = nn.Linear(N*p, N, bias=use_bias)
        else:
            self.fc = nn.Linear(N*self.L, N, bias=use_bias)
        self.fun = fun
        self.bias = None
        self.clamp = clamp
        ##################################  NOTE  #####################################
        # define the function that samples from a Poisson RV with different rates
        # Since the rates should be predefined in torch.distributions.poisson
        # a new poisson distribution object should be created at each time sample
        # using this class. Therefore numpy.random.poisson is being used currently.
        # Should find a better solution by creating the object and 
        # changing only the rate and doing everything in torch. Look at the source code
        # later: https://pytorch.org/docs/stable/_modules/torch/distributions/poisson.html#Poisson
        self.sample = lambda x: torch.tensor(np.random.poisson(x.detach().cpu().numpy()), dtype=torch.float).to(device)
        
    def forward(self, x):
        x = self.fc(x)
        if self.fun is not None:
            x = self.fun(x)
        if self.clamp is not None:
            x = torch.clamp(x, min=0, max=self.clamp)
        return x
    
                
    def fit(self, X, step=1, lam=0.0002, epochs=50, batch_size=2000, method='Poisson', optimization_alg='SGD', lr=5e-2, 
              gamma=0.85, log_interval=100, proximal=False,**kwargs):
        AR_torch_base.fit(self, X, step=step, lam=lam, epochs=epochs, batch_size=batch_size, 
                          method=method, optimization_alg=optimization_alg, lr=lr, 
                          gamma=gamma, log_interval=log_interval, proximal=proximal,**kwargs)

        self.mat = list(self.fc.parameters())[0].detach().cpu().numpy().reshape((self.N, self.N, self.mat_size))
        if self.use_bias:
            self.bias = list(self.fc.parameters())[1].detach().cpu().numpy()
            
            
class GAR_torch_regressor(AR_torch_base):
    def __init__(self, N, p, fun=None, var= 1, use_bias=True, D=None, device='cpu'):
        """
        Gaussian Autoregressive (GAR) process estimator. It estimates parameters 
        of a multivariate GAR.

        Parameters
        ----------
        N : int
            dimension of the Bernoulli process.
        p : int
            time lag. How many samples before sample t should be considered
            for estimating the mean paramters at time t.
        fun : function 
            The inverse link function of the process implemented in torch. The default is None.
        var: float
            variance of the Gaussian inoovation noise. Default is 1.
        use_bias : boolean, optional
            Whether to use bias or not. The default is True.
        device : optional
            which device to save the model in. Use 'gpu{i}' if Cuda is available. 
            The default is 'cpu'.

        Returns
        -------
        None.

        """
        AR_torch_base.__init__(self, N, p, device)
        self.use_bias = use_bias
        if self.L is None:
            self.fc = nn.Linear(N*p, N, bias=use_bias)
        else:
            self.fc = nn.Linear(N*self.L, N, bias=use_bias)
        self.fun = fun
        self.bias = None
        self.var = torch.tensor(var, dtype=torch.float).to(device)
        self.sample = lambda x: x + (torch.sqrt(self.var)*torch.randn(x.shape)).to(self.device)
        
    def forward(self, x):
        x = self.fc(x)
        if self.fun is not None:
            x = self.fun(x)
        return x
    
                
    def fit(self, X, step=1, lam=0.0002, epochs=50, batch_size=2000, method='Poisson', optimization_alg='SGD', lr=5e-2, 
              gamma=0.85, log_interval=100, proximal=False,**kwargs):
        AR_torch_base.fit(self, X, step=step, lam=lam, epochs=epochs, batch_size=batch_size, 
                          method=method, optimization_alg=optimization_alg, lr=lr, 
                          gamma=gamma, log_interval=log_interval, proximal=proximal,**kwargs)

        self.mat = list(self.fc.parameters())[0].detach().cpu().numpy().reshape((self.N, self.N, self.mat_size))
        if self.use_bias:
            self.bias = list(self.fc.parameters())[1].detach().cpu().numpy()
            
class LAR_torch_regressor(AR_torch_base):
    def __init__(self, N, p, fun=None, lam= 1, use_bias=True, D=None, device='cpu'):
        """
        Laplace Autoregressive (LAR) process estimator. It estimates parameters 
        of a multivariate LAR.

        Parameters
        ----------
        N : int
            dimension of the Bernoulli process.
        p : int
            time lag. How many samples before sample t should be considered
            for estimating the mean paramters at time t.
        fun : function 
            The inverse link function of the process implemented in torch. The default is None.
        lam: float
            Laplace process scale. Default is 1.            
        use_bias : boolean, optional
            Whether to use bias or not. The default is True.
        device : optional
            which device to save the model in. Use 'gpu{i}' if Cuda is available. 
            The default is 'cpu'.

        Returns
        -------
        None.

        """
        AR_torch_base.__init__(self, N, p, D, device)
        self.use_bias = use_bias
        if self.L is None:
            self.fc = nn.Linear(N*p, N, bias=use_bias)
        else:
            self.fc = nn.Linear(N*self.L, N, bias=use_bias)
        self.fun = fun
        self.bias = None
        self.lam = torch.tensor(lam, dtype=torch.float)
        self.Laplace = torch.distributions.laplace.Laplace(0, self.lam)
        self.sample = lambda x: x + (self.Laplace.sample(x.shape)).to(self.device)
        
    def forward(self, x):
        x = self.fc(x)
        if self.fun is not None:
            x = self.fun(x)
        return x
    
                
    def fit(self, X, step=1, lam=0.0002, epochs=50, batch_size=2000, method='Poisson', optimization_alg='SGD', lr=5e-2, 
              gamma=0.85, log_interval=100, proximal=False,**kwargs):
        AR_torch_base.fit(self, X, step=step, lam=lam, epochs=epochs, batch_size=batch_size, 
                          method=method, optimization_alg=optimization_alg, lr=lr, 
                          gamma=gamma, log_interval=log_interval, proximal=proximal,**kwargs)

        self.mat = list(self.fc.parameters())[0].detach().cpu().numpy().reshape((self.N, self.N, self.mat_size))
        if self.use_bias:
            self.bias = list(self.fc.parameters())[1].detach().cpu().numpy()
            
            
class AR_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, N, p, D=None, step=1, dtype=torch.float, device='cpu'):
        """
        Args:
            X:
            N:
            p:
                
        """
        self.X = torch.tensor(X, dtype=dtype).to(device)
        self.N = N
        self.p = p
        if D is None:
            self.use_D = False
            self.D = None
        else:
            self.use_D = True
            self.D = torch.tensor(D, dtype=dtype).to(device)
        self.T = X.shape[1] - p - step + 1
        self.step = step

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.use_D:
            features = self.X[:, idx:idx + self.p]
            features = torch.matmul(features, self.D).flatten()
        else:
            features = self.X[:, idx:idx + self.p].flatten()
        target = self.X[:, idx + self.p + self.step - 1]
        sample = (features, target)

        return sample    

        
            
if __name__ == '__main__':
    N = 5
    p = 10
    L = 3
    D = np.random.normal(scale=1/np.sqrt(L), size=(p, L)) #None 
    sparsity = 0.2
    method = 'logistic'
    optimization_alg = 'RMSprop'
    lam = 0.1
    step = 20
    epochs = 40
    Ttr = 60000
    Tts = 5000 + p + step
    batch_size = 2000
    lr = 0.02
    mat_idx = 0
    clamp = 5
    proximal = False
    use_bias = False
    scikit = True
    pytorch = True
    
    
    BAR = BAR_process(N, p, sparsity, fun='logistic', use_bias=use_bias, D=D)
    BAR.simulate(Ttr)
    theta = BAR.mat[mat_idx]
    Xtr = BAR.X
    BAR.simulate(Tts)
    Xts = BAR.X
    fun = lambda x: 1/(1 + torch.exp(-x))  # lambda x:x #
    
    #%% Estimate
    if scikit:
        regressor_scikit = BAR_regressor(N, p, D)
        regressor_scikit.fit(Xtr, method=method, lam=lam) 
        thetahat_scikit = regressor_scikit.mat[mat_idx]
    if pytorch:
        regressor = BAR_torch_regressor(N, p, fun=fun, use_bias=use_bias, D=D)
        regressor.fit(Xtr, method=method, lam=lam, batch_size=batch_size, lr=lr,
                      epochs=epochs, optimization_alg=optimization_alg, proximal=proximal) 
        thetahat = regressor.mat[mat_idx]
    if scikit:
        print(regressor_scikit.SE(BAR.mat))
    if pytorch:
        print(regressor.SE(BAR.mat))
    if scikit and pytorch:
        print(regressor.SE(regressor_scikit.mat))
        print(f'{np.around(theta, 3)}\n{np.around(thetahat, 3)}\n{np.around(thetahat_scikit, 3)}')
        if use_bias:
            print(f'{BAR.bias}\n{regressor.bias}\n{regressor_scikit.bias}')
            
    #%% Predict
    if scikit:
        Xhat_scikit = regressor_scikit.predict(Xts[:, :-step], step=step)
        acc_scikit = regressor_scikit.score(Xhat_scikit, Xts[:, p:])
        plt.figure()
        plt.plot(acc_scikit)
        plt.xlabel('steps predicted in the future')
        plt.ylabel('accuracy of prediction for each neuron')
        plt.title('Scikit')
        
    if pytorch:
        Xhat_pytorch = regressor.predict(Xts[:, :-step], step=step)
        acc, var = regressor.score(Xhat_pytorch[0], Xts[:, p:])
        plt.figure()
        plt.plot(acc)
        plt.xlabel('steps predicted in the future')
        plt.ylabel('accuracy of prediction for each neuron')
        plt.title('PyTorch')
       
        
    #%% Poisson Process   
    # PAR = PAR_process(N, p, sparsity, clamp=clamp, fun='exp', use_bias=use_bias, D=D)
    # PAR.simulate(Ttr)
    # theta = PAR.mat[mat_idx]
    # Xtr = PAR.X
    # PAR.simulate(Tts)
    # Xts = PAR.X
    # fun = lambda x: torch.exp(x) # lambda x:x #
    
    # #%% Estimate
    # if pytorch:
    #     regressor = PAR_torch_regressor(N, p, fun=fun, clamp=clamp, use_bias=use_bias, D=D)
    #     regressor.fit(Xtr, method=method, lam=lam, batch_size=batch_size, lr=lr,
    #                   epochs=epochs, proximal=proximal) 
    #     thetahat = regressor.mat[mat_idx]
    # if pytorch:
    #     print(regressor.SE(PAR.mat))

            
    # #%% Predict
        
    # if pytorch:
    #     Xhat_pytorch = regressor.predict(Xts[:, :-step], step=step)
    #     acc, var = regressor.score(Xhat_pytorch[0], Xts[:, p:],metric='mae')
    #     plt.figure()
    #     plt.plot(acc)
    #     plt.xlabel('steps predicted in the future')
    #     plt.ylabel('accuracy of prediction for each neuron')
    #     plt.title('PyTorch')