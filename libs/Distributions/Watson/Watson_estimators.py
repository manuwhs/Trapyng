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

def get_Watson_muKappa_ML(X, rk = None, parameters = None):
    """ 
    This function efficiently computes the parameters: (mu, kappa),
    for a single Waton distribution, given the weight vectors rk from the EM algorithm.
    If the estimation is ill-posed (degenerated cluster), then it will trigger a RuntimeError
    which will be handled later to deal with the degenerated cluster and take
    an action, like deleting the cluster or changing its parameters.
    """
    
    Niter = parameters["Num_Newton_iterations"]
    try:
        # We compute the Weighted correlation matrix for the component
        # Sk = (N*pimix_k) * B_ik (*) X*X.T
        # If No rk specified, it is just one 
        N,D = X.shape
        if(type(rk) == type(None)):
            rk = np.ones((N,1))*(1/float(N))
        if (1):
            # For the combination of Gaussian and Watson, we need to do the preprocessing here
            X = gf.remove_module (X)
        #################### Get the Mus from here !!
        Sk, EigenValues,V = get_eigenDV_ML(X, rk = rk)
        
        # Obtain the highest and smallest eigenvalue and associated eigenvectors
        d_max = np.argmax(EigenValues)
        d_min = np.argmin(EigenValues)
        mu_pos = V[:,d_max]
        mu_neg = V[:,d_min]
        
        ## We solve the positive and the negative situations and output the one with
        ## the highest likelihood
        eigenValue_pos = EigenValues[d_max]
        eigenValue_min = EigenValues[d_min]
        
        # Compute the explained variance of projections.
        r_neg = np.dot(mu_neg.T,Sk).dot(mu_neg)
        r_pos = np.dot(mu_pos.T,Sk).dot(mu_pos)
    
        
        """ 
        If the explained variance of r_neg is too low, close to 0, then it is illposed
        and we cannot compute it. We choose the other one directly.
        If the explained variance of r_pos is too high, close to 1, then it is also illposed
        and we cannot compute it. We choose the other one directly.
        If both can be computed, then we choose the one with highest likelihood.
        If None can be computed then we have a degenerated cluster and we create the exeption.
        """
        
        tolerance =  1e-3
        
        if(parameters["Allow_negative_kappa"] == "yes"):
            if (r_neg < tolerance and r_pos > 1-tolerance):
                # Case where we have a degenerated cluster
                raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', 
                                   "Degenerated_Cluster_Error",np.sum(rk)/N)
            
            elif (r_neg < tolerance and r_pos < 1-tolerance):
                # Case where the negative kappa case is very unilikely
                kappa_pos = get_Watson_kappa_ML(X, mu_pos,  Sk = Sk, rk = rk, Niter = Niter)
                kappa = kappa_pos
                mu = mu_pos
            elif (r_neg > tolerance and r_pos > 1-tolerance):
                # Case where the positive kappa case is very unilikely
                kappa_neg = get_Watson_kappa_ML(X, mu_neg,  Sk = Sk, rk = rk, Niter = Niter)
                kappa = kappa_neg
                mu = mu_neg
            else:
                # Case where both are possible.
                kappa_pos = get_Watson_kappa_ML(X, mu_pos,  Sk = Sk, rk = rk, Niter = Niter)
                kappa_neg = get_Watson_kappa_ML(X, mu_neg,  Sk = Sk, rk = rk, Niter = Niter)
                likelihood_pos = np.sum(Wad.Watson_pdf_log(X.T,[mu_pos,kappa_pos])*rk.T)
                likelihood_neg = np.sum(Wad.Watson_pdf_log(X.T,[mu_neg,kappa_neg])*rk.T)

            ## Check that there are no duplicated eigenvalues
                if (likelihood_pos > likelihood_neg):
                    if (EigenValues[0] == EigenValues[1]):
                        print ("Warning: Eigenvalue1 = EigenValue2 in MLmean estimation")
                    kappa = kappa_pos
                    mu = mu_pos
                else:
                    if (EigenValues[-1] == EigenValues[-2]):
                        print ("Warning: Eigenvalue1 = EigenValue2 in MLmean estimation")
                    kappa = kappa_neg
                    mu = mu_neg
        else:
            if ( r_pos > 1-tolerance):
                # Case where we have a degenerated cluster
                raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', 
                                   "Degenerated_Cluster_Error",np.sum(rk)/N)
            
            else:
                # Case where the negative kappa case is very unilikely
                kappa_pos = get_Watson_kappa_ML(X, mu_pos,  Sk = Sk, rk = rk, Niter = Niter)
                kappa = kappa_pos
                mu = mu_pos
                if (EigenValues[0] == EigenValues[1]):
                    print ("Warning: Eigenvalue1 = EigenValue2 in MLmean estimation")
                kappa = kappa_pos
                mu = mu_pos
                
        theta = [mu.reshape(D,1), kappa.reshape(1,1)]
            
    except RuntimeError as err:
        theta = None
            
    return theta

# This function obtains the correlation matrix and its eigenvector and values
def get_eigenDV_ML(X, rk = None):
    N,D = X.shape
    if(type(rk) == type(None)):
        rk = np.ones((N,1))/float(N)
        
#    print (X*rk).shape
    Sk = np.dot(X.T,X*rk)   # Covariance matrix. Each is (D,1)*(1*D)
#    unbiased_factor = np.sum(rk)  - (np.sum(rk ** 2)/np.sum(rk) )
    sum_rk = np.sum(rk) 
    
    if (sum_rk < 0.0001):
        # Case where we have a degenerated cluster
    #        print "Degenerated cluster"
        raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', 
                           "Degenerated_Cluster_Error",np.sum(rk)/N)
                
    Sk = Sk/np.sum(rk)  #  /unbiased_factor  # np.sum(rk)            # Not really necesarry

    # Get eigenvalues to obtain the mu
    try:
        # Maybe rk is very small and this fails
        D,V = np.linalg.eig(Sk) # Obtain eigenvalues D and vectors V
    
        # Sometimes this gives also imaginaty partes 
        # TODO: Check what causes the imaginarity
        D = D.astype(float)
        V = V.astype(float)
    except np.linalg.linalg.LinAlgError:
        print ("Sk has failed to have good parameters ")
        raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', 
                           "Degenerated_Cluster_Error",np.sum(rk)/N)

        
    return Sk, D,V #  mu_pos, mu_neg

# This function obtains the positive and negative mus
def get_Watson_mus_ML(X, rk = None):
    Sk, D,V = get_eigenDV_ML(X, rk = rk)
    # Solve this thing
    d_max = np.argmax(D)
    d_min = np.argmin(D)
    mu_pos = V[:,d_max]
    mu_neg = V[:,d_min]
    return  mu_pos, mu_neg
    
def get_Watson_kappa_ML(X, mu,  Sk = None, rk = None, Niter = None):
    """
    This function gets the estimated kappa of the Watson distribution 
    It first computes an approximation and then uses Newton's method.
    Note that even though we could have obtained a valid mu,
    its corresponding kappa might not be tractable since we have to 
    compute its normalizing constant which could not be computable.
    
    The Newtons method implemented is the same as the optimized in the paper
    where we only need to compute the kummer function twice.
    """
    
    n,d = X.shape
    a = 0.5
    c = float(d)/2
    
    if (type(Sk) == type(None)):
        Sk = np.dot(X.T,rk*X)   # Correlation, we weight the samples by the cluster responsabilities r[:,k]
        Sk = Sk/(np.sum(rk))
    
    # Variance explained by the direction
    r = np.dot(mu.T,Sk).dot(mu)
    
    # General aproximation !!!
    if (r > 0.9):    # When r -> 1 
        r = r - 1e-30  # To avoid divide by 0
        BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
    elif(r < 0.1):    # When r -> 0
        r = r + 1e-30  # To avoid divide by 0
        BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    else:            # General approximation
        BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))

#    BGG_opt = newton(get_kappaNewton, BGG, args=([d,r],))
    BGG_opt = Newton_kappa_log(BGG,d,r,Niter)
    return BGG_opt

## Own implementation of the Newton method
def Newton_kappa_log(kappa0,Ndim,r, Niter = None):
    kappa = kappa0
    a = 0.5
    c = float(Ndim)/2
    for i in range(Niter):
        
        M_log = Wad.kummer_log(a, c,kappa)
        Mplus_log = Wad.kummer_log(a + 1, c +1,kappa)
        dM_log = np.log((a/c)) + Mplus_log
    
        # Obtain kappa in natural units
        g = np.exp(dM_log - M_log)
        dg =  (1 - c/kappa)*g + (a/kappa) - g*g
        kappa = kappa - (g - r)/dg

    return kappa

## Function to be used if we use the implementation of the Newton from the optimize library
def get_kappaNewton(k, args):  # The r is computed outsite
    Ndim = args[0]
    r = args[1]
    
    a = 0.5
    c = float(Ndim)/2
    
    M_log = Wad.kummer_log(a, c,k)
    Mplus_log = Wad.kummer_log(a + 1, c +1,k)
    dM_log = np.log((a/c)) + Mplus_log

    g = np.exp(dM_log - M_log)

    return g - r
    
