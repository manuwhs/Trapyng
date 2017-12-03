
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
import matplotlib.colors as ColCon
from scipy import spatial
import datetime as dt
from sklearn import linear_model
import utilities_lib as ul
from graph_lib import gl


from scipy import spatial
from sklearn import preprocessing
from scipy.optimize import fmin
    
# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]

def compute_Kernel(X1, X2, params = {}):
    
#    print X1.shape, X2.shape
    dist = spatial.distance.cdist(X1,X2,'euclidean')
    l = params["l"]
    sigma_0 = params["sigma_0"]
    K = (sigma_0**2) * np.exp(-np.power(dist,2)/(2*l))
    return K
    
class GaussianProcessRegressor(object):
    """
    Implements a GP with mean zero and a custom kernel
    """
    def __init__(self, kernel = None, 
                 sigma_eps = 0.00001, x = None, y = None, 
                 ws = -1, params = None):
        """
        Initialize the GP with the given kernel and a noise parameter for the variance
        Optionally initialize this GP with given X and Y
 
        :param kernel: kernel function, has to be an instance of Kernel
        :param noise_variance:
        :param x: given input data
        :param y: given input label
        :param ws: the size of the window if we only want to use the past ws samples.
        :return:
        """
        self.X = x
        self.Y = y
        # Function to compute the kernel, we 
        self.kernel = kernel
        self.kernel_params = params 
        self.Cov_eps_params = dict([["sigma_eps",sigma_eps]])
        # Covatiance matrix of the noise, if 1 value given, we assume it is k
        self.Cov_eps = None
        # Covariance matrix of the posterior
        self.CovPost = None
        # Kernel values to be used between the training samples
        self.K = None
        # Window of samples from which we compute the posterior.
        self.ws = ws 
        
        # Scaler to preprocess the X variables
        self.Xscaler = None
    
    def get_Cov_eps(self, params = {}, X = None):
        # This function gives the covariance matrix for the given X and parameters
        # In the future it can be extended to a correlated noise given a function.
        
       
        # Preprocess the sigma noise
        Ns,Nd = X.shape
         # Case of uncorrelated shit.
        if "sigma_eps" in params:
             Cov_eps = np.eye(Ns) * (params["sigma_eps"] **2)
        # Case we are given the Correlation matrix directly
        elif "Cov_eps" in params:
             Cov_eps = params["Cov_eps"]

        return Cov_eps
        
    def fit (self, X, y, kernel = None , params = None, sigma_eps = 0):
        """
        We can pass the parameters to the object with this function.
        This function will compute the kernel if needed. 
        """
        
        Ns,Nd = X.shape
        
        self.Xscaler = preprocessing.StandardScaler().fit(X)
        self.X = self.Xscaler.transform(X)
        
        self.Cov_eps_params =  dict([["sigma_eps",sigma_eps]])
        self.Cov_eps = self.get_Cov_eps(params = self.Cov_eps_params, X = self.X)
        self.kernel = kernel 
        self.kernel_params = params
        self.K = kernel(self.X,self.X,params) 
        self.Y = y
        
    def predict(self, X):
        """
        Given data in x, give the mean and covariance of the posterior predictive distribution p(f*|X*, X, f)
        If y is given, the function gives the predicts, as well as update the GP internally,
        including the samples to the model from begining to end.
 
        x should have size N x d1, y of size N x d2, where N is the number of samples
 
        :param x: the input data
        :param y: optional. If given, the GP parameters will be updated
        :return: a tuple (mu, cov, s):
            - mu: the mean of the posterior predictive distribution, of size N x d1
            - cov: the covariance matrix of the posterior predictive distribution, of size N x N
            - s: the standard deviation vector, convenient for plotting. Of size N x 1
        """
        
        Xpred = self.Xscaler.transform(X)
        # covariance of the new data
        K_ss = self.kernel(Xpred, Xpred, self.kernel_params)
        K_s = self.kernel(Xpred, self.X, self.kernel_params)
        
        L = np.linalg.inv(self.K + self.Cov_eps)
        mu = K_s.dot(L).dot(self.Y) 
        Cov = K_ss - K_s.dot(L).dot(K_s.T) # + self.Sigma_noise
                                                                     
#        if y is not None:
#            self.X = np.vstack((self.X, Xpred))
#            self.Y = np.vstack((self.Y, ypred))
#            self.cov = np.hstack((self.cov, K_s))
#            self.cov = np.vstack((self.cov, np.hstack((K_s.T, K_ss))))
 
        return mu, Cov

    def OneStepWindowedPrediction(self, X, Y, ws = 10, lag = 1):
        """
        Once the parameters are set, lets define a Window size,
        then we:
            - Use a training the first ws smaples and 
            - try to estimate the following.
            - Move the window and repeat the process.
            - We predict the last sample + lag
        """
        
        X = self.Xscaler.transform(X)
        Ns, Nd = X.shape
        Npredictions = Ns - ws -1  # Number of predicitons we can do.
        
        mus = np.zeros((Npredictions,1))
        Covs = np.zeros((Npredictions,1))
        nlls = np.zeros((Npredictions,1)) # Of the predicted shits.
        for i in range(Npredictions):
            Xw = X[i:ws+i,:]  # Windowed X
            Yw = Y[i:ws+i,:]
            
            Xval = X[[ws+i+lag-1],:]  # Windowed X
            Yval = Y[[ws+i+lag-1],:]
            # Now it is like a normal Regression :)
            
            # covariance matrcies data
            K = self.kernel(Xw, Xw, self.kernel_params)
            K_ss = self.kernel(Xval, Xval, self.kernel_params)
            K_s = self.kernel(Xval, Xw, self.kernel_params)
            
            Cov_eps = self.get_Cov_eps(self.Cov_eps_params,Xw)
            L = np.linalg.inv(K + Cov_eps)
            mu = K_s.dot(L).dot(Yw) 
            Cov = K_ss - K_s.dot(L).dot(K_s.T) # + self.Sigma_noise
                                                                     
            mus[i,0] = mu   
            Covs[i,0] = Cov      
            # TODO: Estimate likelihood of jghj datasetjgj, given the model
#            nlls[i,0] = self.negative_ll_func(L,Yw,K + Cov_eps)                    
        return mus, Covs
        
        
#        What if clever people were crazy because they find more complex patters in the universe
#        Sometime they learn noise and chaos, or very unlikely bad events. Get distracted
#        Is information the paterns ? More complex systems need less samples of the pattern to reproduce it 
        
    def generate_process(self, X, N = 1, noise = True):
        """ This function will generate a Random Execution of the shit.
            It computes the posterior distribution of the data points and draws samples from it.
        """
        Ns,Nd= X.shape
        mu_s, Cov_s = self.predict(X)
        
        # If we want a realization of the observation y = f(x) + noise
        # then we give it. For simple independent noise is easier. 
        # If the noise is correlated then, the funciton to compute such correlation
        # has to be given.
        # This is different to just giving the predicition (mean) and the variance of the predictions.
        # Since the samples are actually correlated, once we know the value of one, it affects the
        # value of the surroundings :). Given by the posterior covariance funciton of course :)
        
         # This small noise is added to ensure positive defniity matrix
        Cov_s = Cov_s + 1e-10 * np.eye(Ns) 
       
        if (noise):
            Cov_s = Cov_s + np.eye(Ns) * self.Cov_eps[0,0]
        
#        f_s = np.random.multivariate_normal(mu_s.flatten(),Cov_s, size = N)
#        f_s = f_s.T
        # We could have generated the process ourselved by  
        
        L = np.linalg.cholesky(Cov_s)
        f_s = L.dot(np.random.randn(Ns,N)) + mu_s
        
        # return Ns x N
        return f_s

    def negative_ll(self, params ,*args):
        """ This funcion gives us the negative_ll of the data given the model"""
        
        X,Y = args
        sigma_0, l, sigma_eps = params
#        X = self.X
#        Y = self.Y
#        print params
        Ns,Nd = X.shape

        params = dict([["l",l],["sigma_0",sigma_0]])
        K = compute_Kernel(X,X,params = params) 
        Cov_eps = self.get_Cov_eps( params = dict([["sigma_eps",sigma_eps]]), X = X)
        K_reg = K + Cov_eps 
        L = np.linalg.inv(K_reg)
        
#        caca = self.K + self.Cov_eps  
#        print np.linalg.det(caca)
#        print caca
        nll = self.negative_ll_func(L,Y,K_reg)
#        print nll
        return nll
    
    def negative_ll_func(self,L,Y,K_reg):
        # Given (X,Y), we compute its private K and then feed this shit
        Ns = Y.shape[0]
        nll = .5 * Y.T.dot(L).dot(Y) 
        nll +=  .5 * np.log(np.linalg.det(K_reg)) 
        nll += .5 * Ns * np.log(2*np.pi)
        return nll
        
    def optimize_parameters(self, sigma_0, l, sigma_eps):
        """ This function should optimize the parameters """
        # We give as intial guess the desired values
        xopt = fmin(func= self.negative_ll, 
                    x0 = np.array([sigma_0, l, sigma_eps]), 
                    args=(self.X,self.Y))    
        
        return xopt