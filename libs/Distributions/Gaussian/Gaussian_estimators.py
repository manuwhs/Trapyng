# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:47:19 2017

@author: montoya
"""

import import_folders
from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.optimize import newton
import numpy as np
import utilities_lib as ul

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf
import warnings

################################################################################
################## Mother function #############################################
def get_Gaussian_muSigma_ML(X, rk = None, parameters = None):
    N,D = X.shape
    
    if(type(rk) == type(None)):
        rk = np.ones((N,1))*(1/float(N))
        
    sum_rk  = np.sum(rk)
    if (sum_rk < 0.0001):
        # Case where we have a degenerated cluster
    #        print "Degenerated cluster"
        raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', "Degenerated_Cluster_Error",np.sum(rk)/N)
                
    unbiased_factor = sum_rk - (np.sum(rk ** 2)/sum_rk)
    

    ## Get mean
    muk = np.sum(X*rk, axis = 0)/sum_rk
    muk = muk.reshape(1,D)
#    muk = np.zeros(muk.shape)
#    print ("rk_shape")
#    print (rk.shape)
#    print ("mu_shape")
#    print (muk.shape)
#    print (sum_rk)
#    
    ## Get covariance
    
    if (type(parameters) == type(None)):
        Sigma_type = "full"
    else:
        Sigma_type = parameters["Sigma"]
    if (Sigma_type == "full"):
        Sk = np.dot((X - muk).T,(X - muk)*rk)   # Covariance matri. Each is (D,1)*(1*D)
        Sk = Sk/unbiased_factor  # np.sum(rk)            # Not really necesarry
    elif(Sigma_type == "diagonal"):
        ## TODO: More efficient
        sigmas = [] 
        for d in range(D):
          sigmas.append(np.dot((X[:,[d]] - muk[0,d]).T,(X[:,[d]] - muk[0,d])*rk) )        
        sigmas = np.diag(np.array(sigmas).flatten())
        Sk = sigmas/unbiased_factor
        
    theta = [muk.reshape(D,1), Sk]
    return theta