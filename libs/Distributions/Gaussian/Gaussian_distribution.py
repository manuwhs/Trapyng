# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:47:19 2017

@author: montoya
"""

import import_folders
from scipy.special import hyp1f1 as scipy_hyp1f1
import mpmath
from scipy.special import gamma
from scipy.optimize import newton
import numpy as np
import utilities_lib as ul

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

import warnings
import mpmath as mpm

from scipy.linalg import cholesky

def Gaussian_K_pdf_log (X, theta, Cs_log = None, parameters = None):
    # Extension of Watson_pdf_log in which we also accept several clusters
    # We have to be more restrict in this case and the parameters must be:
    # X (D,Nsamples)  mu(D,K) kappa(K) cp_log(K)
    # The result is (Nsamples, K)
    D, N = X.shape
    K = len(theta)
    
    if (type(parameters) == type(None)):
        diagonal = False
    else:
        if (parameters["Sigma"] == "full"):
            diagonal = False
        else:
            diagonal = True
        
    log_pdfs = []
    
    if (type(Cs_log) == type(None)):
        Cs_log = [None]*K
        
    for k in range(K):
        # TODO: Play with norm constant
        log_pdf_i = Gaussian_pdf_log(X, theta[k], diagonal = diagonal, Cs_log = Cs_log[k])
        log_pdfs.append(log_pdf_i)
    
    log_pdfs = np.concatenate(log_pdfs, axis = 1)
    
    return log_pdfs

def Gaussian_pdf_log (X, theta_k, Cs_log = None, diagonal = False):
    # Extension of Watson_pdf_log in which we also accept several clusters
    # We have to be more restrict in this case and the parameters must be:
    # X (D,Nsamples)  mu(D,K) kappa(K) cp_log(K)
    # The result is (Nsamples, K)
    D, N = X.shape
    
#    print K
#    print "theta_len", len(theta)
#    print "theta0", theta[0][0].shape, theta[0][1].shape    
    ########### Obtain parameters from theta list ############
    mu = theta_k[0]
    Sigma = theta_k[1]
    
    cp_log_k = get_cp_log(Sigma, diagonal)
    Cs_log = np.array(cp_log_k)
    Cs_log = Cs_log.reshape(Cs_log.size,1)
    
    ####### Compute the inverse of the Cov matrix ##############
    if (diagonal):
#        print np.diagonal(Sigma)
        Sigma_inv = np.diag(1/np.diagonal(Sigma + 1e-100*np.eye(D)))
#        print "diagonal"
    else:
        Sigma_inv = np.linalg.inv(Sigma)
    aux1 = X - mu
    
#    print aux1.shape
    ## Efficient computation of the likelihood for many samples
    log_pdf = aux1.T.dot(Sigma_inv)*(aux1.T)
    log_pdf = np.sum(log_pdf, axis =1)
#    print (log_pdf.shape)
    
    log_pdf =  - log_pdf/2 + Cs_log
    
    return log_pdf.T
    

def get_cp_log(Sigma, diagonal = False):
    D= Sigma.shape[0]
    
    if (diagonal == False):
        try:
#            L = cholesky(Sigma)
#            log_det = np.sum(2*np.log(np.diag(L)))
            det_K = mpm.det(Sigma + 1e-100*np.eye(D))
    #        print(det_K)
            log_det = float(mpm.log(det_K))
    #        print (log_det)
    #        det_K = np.linalg.det(K[:N_det,:N_det]+ 1e-10*np.eye(N_det))   # Determinant ! "Noisiness of the kernel"

        except RuntimeError as err: 
            return None
    else:
        log_det = np.sum(np.log(np.diagonal(Sigma+ 1e-100*np.eye(D))))
#        print "Gaussian fet"
    cp_log = -(1/2.0)*(D*np.log(2*np.pi) +  log_det)
    
#    print (cp_log)
    return cp_log

def get_Cs_log(theta, parameters = None):
    Sigma = theta[1]
    if (type(parameters) == type(None)):
        diagonal = False
    else:
        if (parameters["Sigma"] == "full"):
            diagonal = False
        else:
            diagonal = True
        
    return get_cp_log(Sigma, diagonal )


def init_params(X,K ,theta_init = None, parameters = None):
    # Here we will initialize the  theta parameters of the mixture model
    # THETA PARAMETERS OF THE K COMPONENTS
    # Give random values to the theta parameters (parameters of the distribution)
    # In this case the parameters are theta = (mu, kappa)
    
    # We need at least the number of clusters K and the dimensionality of the
    # distirbutoin D.
    N,D = X.shape
    if (type(theta_init) == type(None)): # If not given an initialization
        mus = np.random.randn(D,K)* parameters["mu_variance"]    # np.sqrt(parameters["mu_variance"]);
#        print mus
        Sigma_min = parameters["Sigma_min_init"]
        Sigma_max =   parameters["Sigma_max_init"]

        ## The covariance matrices initialized are Diagonal
        
        Sigmas = []
        for k in range(K):
           diag_Sigma = np.random.uniform(Sigma_min,Sigma_max,D) 
           Sigma_k = np.diag(diag_Sigma)
           Sigmas.append(Sigma_k)
           
        ####### Put theta in the right format ###########
        theta = []
        for k in range(K):
            theta.append([mus[:,[k]],Sigmas[k]])
        return  theta
                
    else:
        return theta_init

########################################################################################
#################### Special functions for the EM ######################################
########################################################################################


def degenerated_estimation_handler(X, rk , prev_theta_k, parameters = None ):
    """ Function to compute another parameters for the cluster since 
    we cannot compute the new ones because it is degenerated.
    We might need the previous_theta_k for our algorithm and some hyperparameters.
    We might choose to set it to the previous hyperparameters for example or saturate something."""
    
    # In this case, we just put the previous parameters, or saturare the previous as well
    Sigma_min = parameters["Sigma_min_singularity"]
    mu_k = prev_theta_k[0]
    kappa_k = prev_theta_k[1]
    
    ####### Saturation case ################
#    kappa_k = np.sign(prev_theta_k[1])*Kappa_max
    return [mu_k, kappa_k]

def degenerated_params_handler(X, rk , prev_theta_k, parameters = None):
    """ Function to compute another parameters for the cluster since
    the ones we have now make it intractable to compute it.
    For example because we cannot compute the normalization constant with the 
    given set of parameters"""
    
    
    ## TODO: Not used now 
    # parameters["Sigma_min_pdf"] 
    Sigma_min = parameters["Sigma_min_init"]
    Sigma_max =   parameters["Sigma_max_init"]
    D = prev_theta_k[1].shape[0]
    ### We generate a diagonal covariance matrix in that point !
    diag_Sigma = np.random.uniform(Sigma_min,Sigma_max,D) 
    Sigma_k = np.diag(diag_Sigma)

    mu_k = prev_theta_k[0]
#    new_mu_pos, new_mu_neg = Wae.get_Watson_mus_ML(X, rk)
#    if (theta[1][0,k] >= 0):
#        new_mu = new_mu_pos
#    else:
#        new_mu = new_mu_neg
#    kappas[:,k] = Kappa_max # * kappas[:,k]/np.abs(kappas[:,k])
    
    return [mu_k, Sigma_k]

