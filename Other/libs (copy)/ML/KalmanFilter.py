
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
from scipy.optimize import fmin
import copy

class KalmanFilter(object):
    """
    Implements a GP with mean zero and a custom kernel
    """
    def __init__(self,
             A = None, B = None, C = None, # System Dynamics
             SigmaXXpred0 = None, Xpred0 = None, # Initial prediction
             SystemCovNoise = None, MeasurCovNoise = None):  # System and Measuring noise
        """
        Initialize the GP with the given kernel and a noise parameter for the variance
        Optionally initialize this GP with given X and Y
        
        # System of equations A
        #   Xt = A*Xt-1 + B*Ut + SystemNoise  
        #   Yt = Cx_k  + MeasurementNoise
        # In the normal formulation Xt is a column vector !!
        """
        self.A,self.B,self.C = A,B,C 
        self.SigmaXXpred0, self.Xpred0 = SigmaXXpred0, Xpred0
        self.SystemCovNoise, self.MeasurCovNoise = SystemCovNoise, MeasurCovNoise 

    def init_KF(self, y):
        ####################### Initialize Kalman Filter ##########################
        ## Time to choose our priors of prediction !!
        ## Important: 
        #      - Variables with "hat" are estimated variables (V(t|t))
        #      - Variables with "pred" are predicted variables (V(t+1|t))
        # In the begining we have to define Xpred(1|0).
        # Observed samples start at time t = 1. t = 0 is our prior thingy.
        
        ## We just initialize the parameters of the kalman it with a good guess. 
        # In this case we just use the initial obervation of Yo as mean of the prior. X0hat = Y0
        Nd,Ns = y.shape
        Xpred0 = y[:,[0]] # Initilization Xpred for X1 given Y0
        
        # Initilize the uncertainty of the first samples high enough if we are not sure
        # or low enough if we are sure of out initial Xhat
        # As more samples arise, this influence of this initial decision will be decrease.
        # We choose a small uncertainty of the initializatoin
        varpred0 = 20
        SigmaXXpred0  = np.eye(Nd) * varpred0
        return Xpred0,SigmaXXpred0
    
    def init_fit_param(self, X, y,A, B, C, 
             SigmaXXpred0, Xpred0,SystemCovNoise, MeasurCovNoise):
        # This funciton initializices properly the parameters when callinga fit.
        ## In this part we just select between the ones given and the inner ones
        if (type(A) == type(None)):
            A = self.A
        if (type(B) == type(None)):
            B = self.B
        if (type(C) == type(None)):
            C = self.C
            
        if (type(SigmaXXpred0) == type(None)):
            SigmaXXpred0 = self.SigmaXXpred0
        if (type(Xpred0) == type(None)):
            Xpred0 = self.Xpred0
            
        if (type(SystemCovNoise) == type(None)):
            SystemCovNoise = self.SystemCovNoise
        if (type(MeasurCovNoise) == type(None)):
            MeasurCovNoise = self.MeasurCovNoise
        
        # If initial conditions are not given, we use
        #   - The initial value as the prediction
        #   - A small variance 
        if (type(Xpred0) == type(None)):
            Xpred0, SigmaXXpred0 = self.init_KF(y)
        
        return A,B,C, Xpred0, SigmaXXpred0, SystemCovNoise, MeasurCovNoise
    
    def fit (self, X, y,                   # Data
             A = None, B = None, C = None, # System Dynamics
             SigmaXXpred0 = None, Xpred0 = None, # Initial prediction
             SystemCovNoise = None, MeasurCovNoise = None):  # System and Measuring noise
        """
        This function runs the Kalman filter once over the data given
        and returns the data
        """
        self.X = X
        self.Y = y
        # Now the magic starts :)
        Ns,Nd = y.shape
        y = y.T # So we need to transpose our normal X[Nsam,Ndim]

        A,B,C, Xpred0, SigmaXXpred0, SystemCovNoise, MeasurCovNoise = self.init_fit_param( X, y,A, B, C, 
             SigmaXXpred0, Xpred0,SystemCovNoise, MeasurCovNoise)
            
        ######### Initialize variables just to keep intermediate results ##########
        XhatList = []               # Store here the estimations of X
        SigmaXXhatList = []        # Store here the estimated Sigma of X
        XpredList = []              # Store here the predictions of X
        SigmaXXpredList = []        # Store here the predicted Sigma of X
        
        XpredList.append(Xpred0)           # Store here the predictions of X
        SigmaXXpredList.append(SigmaXXpred0)       # Store here the predicted Sigma of X

        ######## Initialize the first step data  ########
        Xpred = Xpred0
        SigmaXXpred = SigmaXXpred0
        ## Calculate covariance matrix between observarionts and latent variables
        SigmaXYpred  = SigmaXXpred.dot(C.T)
        # Maybe same as the measurement noise would be appropiate.
        SigmaYYpred = C.dot(SigmaXYpred) + MeasurCovNoise
    
        ################## RECONSTRUCTION OF THE STATE and ONE prediction ############################
        for n in range (0,Ns):
          #############   ESTIMATING STEP  ###################
          # Estimation of theparameters ("hat" variables V(t|t))
          # Calculated from the predicted values and the new sample
          K = SigmaXYpred.dot(np.linalg.inv(SigmaYYpred))  # Kalman Gain
          XpredError = y[:,[n]]- Xpred
          Xhat = Xpred + K.dot(C.dot(XpredError))
          SigmaXXhat = SigmaXXpred - K.dot(SigmaYYpred.dot(K.T))
          #############   PREDICTION STEP  ###################
          ## Predict the variables for the next instance  V(t+1|t)
          Xpred = A.dot(Xhat)  # We use the correlation bit
          SigmaXXpred = A.dot(SigmaXXhat.dot(A.T)) + SystemCovNoise
          SigmaXYpred = SigmaXXpred.dot(C.T)
          SigmaYYpred = C.dot(SigmaXYpred) + MeasurCovNoise
          ################# Storing data  ################
          XhatList.append(Xhat)
          SigmaXXhatList.append(SigmaXXhat)
          XpredList.append(Xpred)
          SigmaXXpredList.append(SigmaXXpred)
      
        ############ Restructure Data in normal np format #######
        XhatList = np.concatenate(XhatList, axis = 1)
        XpredList = np.concatenate(XpredList, axis = 1)
        Yhat =  C.dot(XhatList)
        Ypred = C.dot(XpredList)
    
        Yhat = Yhat.T
        Ypred = Ypred.T
        
        # Set data as local variables
        self.SigmaXXhatList = SigmaXXhatList
        self.SigmaXXpredList = SigmaXXpredList
        self.Yhat = Yhat
        self.Ypred = Ypred
        
        return copy.deepcopy([Yhat,SigmaXXhatList,Ypred, SigmaXXpredList])
    
    def predict(self, Xpredtest0 = None, SigmaXXpredtestList0 = None,
                Ntst = 10):
        """
        This function predicts the future of the Kalman from the last estimation.
           - If param are given, those are used.
           - If not, the local variables are used.
           - If the local variables do not exist, then we should have called fit first
        """
        
        if (type(Xpredtest0) == type(None)):
            Xpredtest = self.Ypred[[-1],:].T
            SigmaXXpredtest = self.SigmaXXpredList[-1]
        else:
            Xpredtest= Xpredtest0
            SigmaXXpredtest = SigmaXXpredtestList0
            
        ## Variables to store the results
        XpredtestList = []
        SigmaXXpredtestList = []
        
        # The first prediction is the one calculated last in the previous loop
        XpredtestList.append( Xpredtest)
        SigmaXXpredtestList.append(SigmaXXpredtest)
        
        # We calculate the rest
        A,C,SystemCovNoise = self.A, self.C, self.SystemCovNoise
        
#        print SigmaXXpredtest.shape
        for n in range(Ntst):
          # Perform future predictions
          Xpredtest = A.dot(Xpredtest)
          SigmaXXpredtest = A.dot(SigmaXXpredtest.dot(A.T)) + SystemCovNoise 
          
          XpredtestList.append(Xpredtest)
          SigmaXXpredtestList.append(SigmaXXpredtest)
         
        ##################  Transform data and preprocess  #########################################v
        XpredtestList = np.concatenate(XpredtestList, axis = 1)
        Ypredtest = C.dot(XpredtestList)
        ## Now we transpose things back to our normal Nsamxdim
        Ypredtest = Ypredtest.T
        # Set data as local variables
        self.Ypredtest = Ypredtest
        self.SigmaXXpredtestList = SigmaXXpredtestList

        return copy.deepcopy([Ypredtest,SigmaXXpredtestList])

        
    def negative_ll(self, params ,*args):
        """ This funcion gives us the negative_ll of the data given the model"""
        
        X,y = args
        varPrice, varPriceDiff, varPriceNoise = params
        SystemCovNoise, MeasurCovNoise = self.build_CovMatrix(y, varPrice, varPriceDiff, varPriceNoise)

#        print SystemCovNoise
        Yhat,SigmaXXhatList,Ypred, SigmaXXpredList = self.fit(X,y, SystemCovNoise = SystemCovNoise, MeasurCovNoise = MeasurCovNoise)
        
        # Compute the YY variance
        C = self.C
        SigmaYYpredList = []
        for i in range(len(SigmaXXpredList)):
            SigmaXXpred = SigmaXXpredList[i]
            SigmaXYpred = SigmaXXpred.dot(C.T)
            SigmaYYpred = C.dot(SigmaXYpred) + MeasurCovNoise
            SigmaYYpredList.append(SigmaYYpred)
        
        nll = self.negative_ll_func(y, Ypred[:-1,:], SigmaYYpredList)

        return nll
    
    def negative_ll_func(self, y, ypred, SigmaYYpredList):
    # Returns the log-likelihood of the likelihood of the prediction given
    # the model and the current state P(Ypred | Ypred)
        nll = 0
        Ns,Ndim = y.shape
        
        yerr = ypred - y  # Difference at each point, between the predicted and the real.
        for i in range (Ns):
            yerr_i = yerr[[i],:]
            nll += np.log(np.linalg.det(SigmaYYpredList[i]))
            nll += (yerr_i).dot(np.linalg.inv(SigmaYYpredList[i])).dot(yerr_i.T)
            
        nll += Ns*np.log(np.pi * 2)
        
        nll = nll/2
        return nll

        
    def optimize_parameters(self, varPrice, varPriceDiff, varPriceNoise):
        
        """ This function should optimize the parameters """
        # We give as intial guess the desired values
        xopt = fmin(func= self.negative_ll, 
                    x0 = np.array([varPrice, varPriceDiff, varPriceNoise]), 
                    args=(self.X,self.Y))    
        
        return xopt
    
    def build_CovMatrix(self, y, varPrice, varPriceDiff,varPriceNoise):
        """ Returns the proper Covariance matrix from a set of parameters """
        Ns,Nd = y.shape
        SystemCovNoise = np.eye(Nd) *varPrice
        SystemCovNoise[1,1] = varPriceDiff
        MeasurCovNoise = np.eye(Nd) *varPriceNoise 
        
        return SystemCovNoise, MeasurCovNoise
        
        