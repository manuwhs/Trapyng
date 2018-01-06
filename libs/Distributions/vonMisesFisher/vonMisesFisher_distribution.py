import numpy as np
from scipy.special import ive,iv
from scipy import pi

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

def vonMisesFisher_K_pdf_log (X, theta, Cs_log = None, parameters = None):
    # Extension of Watson_pdf_log in which we also accept several clusters
    # We have to be more restrict in this case and the parameters must be:
    # X (D,Nsamples)  mu(D,K) kappa(K) cp_log(K)
    # The result is (Nsamples, K)
    D, N = X.shape
    K = len(theta)

    if (1):
        # For the combination of Gaussian and Watson, we need to do the preprocessing here
        X = gf.remove_module (X.T).T
        
    if (type(Cs_log) == type(None)):
        Cs_log = [None]*K
        
    log_pdfs = []
    for k in range(K):
        # TODO: Play with norm constant
        log_pdf_i = vonMisesFisher_pdf_log(X, theta[k],  Cs_log = Cs_log[k])
        log_pdfs.append(log_pdf_i)
    
    log_pdfs = np.concatenate(log_pdfs, axis = 1)
    
    return log_pdfs

def vonMisesFisher_pdf_log(X, theta_k, Cs_log = None):
    
    mu = theta_k[0]
    kappa = theta_k[1]
    
    ######## Make sure the parameters have proper dimensions #######
    mu = np.array(mu)
    mu = mu.flatten().reshape(mu.size,1)
    
    X = np.array(X)
    X = X.reshape(mu.size,X.size/mu.size)
    
    D = mu.size    # Dimensions of the distribution
    N = X.shape[1] # Number of samples to compute the pdf
    
    if (0):
        # For the combination of Gaussian and Watson, we need to do the preprocessing here
        X = gf.remove_module (X.T).T
        
    D,N = X.shape
    X = X.T
    
    kappa = theta_k[1]
    mu = theta_k[0]
    if (type(Cs_log) == type(None)):
        Cs_log = get_cp_log(D,kappa)  
    log_pdf = Cs_log + np.dot(X,mu)*kappa
    
#    if (log_pdf.size == 1): # Turn it into a single number if appropiate
#        log_pdf = float(log_pdf)
        
    return log_pdf


def get_cp_log(p,kappa):
    p = float(p)
    cp = (p/2-1)*np.log(kappa)
#    bessel_func = ive (p/2-1,kappa) * np.exp(kappa)
#    print ".............."
#    print bessel_func
    bessel_func = mpmath.besseli(p/2-1,float(kappa))
#    bessel_func = float(bessel_func)
#    print bessel_func
    bessel_func_log = float(mpmath.log(bessel_func))
#    print bessel_func_log
    cp += -(p/2)*np.log(2*pi)-bessel_func_log
    
    return cp


def get_Cs_log(theta,  parameters = None):
    kappa = theta[1]
    Ndim = theta[0].size
    return get_cp_log(Ndim, kappa)


def init_params(X, K, theta_init = None, parameters = None):
    # Here we will initialize the  theta parameters of the mixture model
    # THETA PARAMETERS OF THE K COMPONENTS
    # Give random values to the theta parameters (parameters of the distribution)
    # In this case the parameters are theta = (mu, kappa)
    
    # We need at least the number of clusters K and the dimensionality of the
    # distirbutoin D.
    N,D = X.shape
    if (type(theta_init) == type(None)): # If not given an initialization
        mus = np.random.randn(D,K);
        mus = gf.normalize_module(mus.T).T
#        print mus

        Kappa_min = parameters["Kappa_min_init"]
        Kappa_max = parameters["Kappa_max_init"]
        
        kappas = np.random.uniform(Kappa_min,Kappa_max,K)
        kappas = kappas.reshape(1,K)
        ####### Put theta in the right format ###########
        theta = []
        for k in range(K):
            theta.append([mus[:,[k]],kappas[:,[k]]])
    else:
        return theta_init

    return  theta
    

########################################################################################
#################### Special functions for the EM ######################################
########################################################################################

def avoid_change_sign_centroids (theta_new, theta_prev):
        ## Improvement for visualizing 
        # Since the polarity of mu does not count, and we dont want it to be changing
        # randomly, we need a way to keep it as most as possible in one side,
        # for that we multiply it with the previous, if pol = +, we keep it, if -, we change

        K = len(theta_new)
        
        if (type(theta_prev) != type(None)):  # Avoid first initialization
            for k in range(K):
                if (type(theta_new[k]) != type(None)):
                    
                    signs = np.sum(np.sign(theta_new[k][0] *  theta_prev[k][0]))
                    if (signs < 0):
                        theta_new[k][0] = -theta_new[k][0]
        return theta_new
    
def degenerated_estimation_handler(X, rk , prev_theta_k, parameters = None ):
    """ Function to compute another parameters for the cluster since 
    we cannot compute the new ones because it is degenerated.
    We might need the previous_theta_k for our algorithm and some hyperparameters.
    We might choose to set it to the previous hyperparameters for example or saturate something."""
    

    Kappa_max = parameters["Kappa_max_singularity"]
    
    # In this case, we just put the previous parameters, or saturare the previous as well
    
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
    
    Kappa_max = parameters["Kappa_max_pdf"]
    
#    if (type(prev_theta_k) != type(None)):
        
    kappa_k = np.sign(prev_theta_k[1])*Kappa_max
    mu_k = prev_theta_k[0]
#    new_mu_pos, new_mu_neg = Wae.get_Watson_mus_ML(X, rk)
#    if (theta[1][0,k] >= 0):
#        new_mu = new_mu_pos
#    else:
#        new_mu = new_mu_neg
#    kappas[:,k] = Kappa_max # * kappas[:,k]/np.abs(kappas[:,k])
    
    return [mu_k, kappa_k]

