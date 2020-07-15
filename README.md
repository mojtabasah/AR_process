# AR process in Pytorch
This project allows you to generate data from autoregressive (AR) processes, and fit an AR process to data. The fitting of AR process is implemented in [scikit-learn](https://scikit-learn.org/stable/) for simple models like Gaussian, Bernoulli, and Poisson AR processes when the maximum likelihood (ML) estimation is used to fit the parameters. But since this setting is very limited, we have implemented the estimation in Pytorch as well. This allows us to fit almost any AR process with any general loss function (negative log-likelihood being an example), using gradient based methods. In each case, there is a base model for generating data and a base model for fitting models, and implementing new AR processes is as simple as defining a sampling function based on the mean paramters of AR process. Similarly, for estimation, implementing estimators for new processes is as simple as implementing negative-log-likelihood loss or even using losses that are already implemented.

## Prerequisites
Other than basic Python libraries like Numpy, Pytorch and Scikit-learn are also required depending on the type of estimator used to fit the data. To generate data, Numpy is enough.

## Using the code
There are three files that can be used independently of each other. [AR_process_model.py](https://github.com/mojtabasah/AR_process/blob/master/AR_process_model.py) is used to simulate data with an AR process. [AR_scikit_regressor.py](https://github.com/mojtabasah/AR_process/blob/master/AR_scikit_regressor.py) implements fitting an AR process to data using Scikit-Learn. And, [AR_torch_regressor.py](https://github.com/mojtabasah/AR_process/blob/master/AR_torch_regressor.py) implements fitting the process to data using Pytorch package. This guide is good enough to get you started on using these files. A Jupyter notebook of examples will be added in near future.

### [AR_process_model.py](https://github.com/mojtabasah/AR_process/blob/master/AR_process_model.py)
There is a base model class in this file that is used to implement different types of AR processes. A few simple models such as Bernoulli (BAR), Gaussian (GAR), Poisson (PAR), and Laplace (LAR) autoregressive models are already implemented. To implement other models, the sampling function of the process given the mean paramter needs to be implemented. See the implementation of BAR, GAR or Poisson process for example. To create a Bernoulli process with $N$ dimensions, $p$ time lags, and $s$-sparse process matrices can be created by:
```
BAR = BAR_process(N=5, p=10, sparsity=0.1)
```
Then, the process can be simulated for a time $T$ using
```
BAR.simulate(T=1000)
data = BAR.X   #This would be an N*T matrix
```
### [AR_scikit_regressor.py](https://github.com/mojtabasah/AR_process/blob/master/AR_scikit_regressor.py)
This file implements regression for simple AR models using Scikit-Learn. Example usage for a BAR process is
```
regressor = BAR_regressor(N=5, p=10)
regressor.fit(data, lam=0.1)
```
where $\lambda$ corresponds to $\ell_1$ regularization (scaled accroding to number of samples and paramters). This would fit the BAR using regularized maximum likelihood.
The paramters can be recovered as follows:
```
bias = regressor.bias
mats = regressor.mat
```

See definition of classes to see what options are available for each method.

### [AR_torch_regressor.py](https://github.com/mojtabasah/AR_process/blob/master/AR_torch_regressor.py)
This file implements the regressors in Pytorch. A base model class is implemented as AR_torch_base. Using this base model, some common AR models are implemented such as BAR, GAR, PAR. and LAR. Implementing new models is as simple as implementing how the mean paramters are generated from previous samples in the forward method, and how sampling is done using mean paramters. See the implementation of preimplemented models to see how it is done. Also, see AR_dataset class implemented in the same file to see how AR data is genreted from a seqence of samples to be used for training and testing in Pytorch models.

Using the regressors in this file are very similar to the ones implemeted in Scikit-learn:
```
regressor = BAR_torch_regressor(N=5, p=10)
regressor.fit(data, lam=0.1)
bias = regressor.bias
mats = regressor.mat
```
See the example at the end of the file to see how all these classes can be used and to see more options that these classes support. For a full list of supported methods and argumets, please refer to the class definitions.
