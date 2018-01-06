# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:47:19 2017

@author: montoya
"""
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



def Watson_K_pdf_log (X, theta, Cs_log = None, parameters = None):
    # Extension of Watson_pdf_log in which we also accept several clusters
    # We have to be more restrict in this case and the parameters must be:
    # X (D,Nsamples)  mu(D,K) kappa(K) cp_log(K)
    # The result is (Nsamples, K)
    D, N = X.shape
    K = len(theta)
    
    ## TODO: better manage this
    if (1):
        # For the combination of Gaussian and Watson, we need to do the preprocessing here
        X = gf.remove_module (X.T).T
#    print K
#    print "theta_len", len(theta)
#    print "theta0", theta[0][0].shape, theta[0][1].shape    
    ########### Obtain parameters from theta list ############
    mus = []
    kappas = []
    for theta_k in theta:
        mus.append(theta_k[0])
        kappas.append(theta_k[1])
    
    mus = np.concatenate(mus, axis = 1)
    kappas = np.array(kappas).reshape(K,1)
    
#    print "kappas", kappas.shape
    
    ## Obtain the coefficients if needed
    if(type(Cs_log) == type(None)):
        Cs_log = []
        for k in range(K):
            cp_log_k = get_cp_log(D, kappas[k,0])
            # If for any of the clusters we cannot compute the normalization constant
            # then we just indicate it
            if (type(cp_log_k) == type(None)):
                return None
            Cs_log.append(cp_log_k)
        
#    print "Cs_log", Cs_log
    ## Perform the computation for several clusters !!
    kappas = np.array(kappas)
    kappas = kappas.reshape(kappas.size,1)
    
    Cs_log = np.array(Cs_log)
    Cs_log = Cs_log.reshape(Cs_log.size,1)
    
    aux1 = np.dot(mus.T, X)
    log_pdf = Cs_log + (kappas * np.power(aux1,2))
    return log_pdf.T

def Watson_pdf_log (X, theta, C_log = None):
    # Compute this in case that the probability is too high or low for just one sample
    # cp is ok, we can calculate normaly and then put it log
    # cp goes to very high if low dimensions and high kappa
    
    # If we indicate cp_log  we do not compute it.
    
    # Watson pdf for a 
    # mu: [mu0 mu1 mu...] D weights of the main direction of the distribution
    # kappa: Dispersion value
    # X: Vectors we want to know the probability for
    
    # Just make sure the matrixes are aligned
    ## TODO: better manage this
    ########### Obtain parameters from theta list ############
    mu = theta[0]
    kappa = theta[1]
    
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
    ######### Check if we can save the computation of norm constant #######
    if (type(C_log) == type(None)):
        C_log = get_cp_log(D, kappa) # Function compute normalization const
    
    ###### Compute the probability ###############
    aux1 = np.dot(mu.T, X)
    log_pdf = C_log + (kappa * np.power(aux1,2))
    
    if (log_pdf.size == 1): # Turn it into a single number if appropiate
        log_pdf = float(log_pdf)
        
    return log_pdf

def get_Cs_log(theta,  parameters = None):
    kappa = theta[1]
    Ndim = theta[0].size
    return get_cp_log(Ndim, kappa)

def get_cp_log(Ndim, kappa):
    gammaValue_log = np.log(gamma(float(Ndim)/2))
    # Confluent hypergeometric function 1F1(a, b; x)
    
    try:
        M_log = kummer_log(0.5, float(Ndim)/2, kappa)   
    except RuntimeError as err:
        return None
    cp_log = gammaValue_log - (np.log(2*np.pi) *(float(Ndim)/2) + M_log)
    return cp_log

def init_params(X, K, theta_init = None, parameters = None):
    # Here we will initialize the  theta parameters of the mixture model
    # THETA PARAMETERS OF THE K COMPONENTS
    # Give random values to the theta parameters (parameters of the distribution)
    # In this case the parameters are theta = (mu, kappa)
    
    # We need at least the number of clusters K and the dimensionality of the
    # distirbutoin D.
#    print X.shape
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

#########################################################################################
############ Normalization constant functions #########################
#########################################################################################
def kummer_log(a,b,x):
    ## First try using the funcion in the library.
    ## If it is 0 or inf then we try to use our own implementation with logs
    ## If it does not converge, then we return None !!
    a = float(a); b = float(b); x = float(x)
    
#    f_scipy = scipy_hyp1f1(a,b,x)
    f_mpmath = mpmath.hyp1f1(a,b,x)
    f_log = float(mpmath.log(f_mpmath))
    if (np.isinf(f_log) == True):
#        warnings.warn("hyp1f1() is 'inf', trying log version,  (a,b,x) = (%f,%f,%f)" %(a,b,x),UserWarning, stacklevel=2)
        f_log = kummer_own_log(a,b,x)
#        print f_log
        
    elif(f_mpmath == 0):
#        warnings.warn("hyp1f1() is '0', trying log version, (a,b,x) = (%f,%f,%f)" %(a,b,x),UserWarning, stacklevel=2)
        raise RuntimeError('Kummer function is 0. Kappa = %f', "Kummer_is_0", x)
#        f_log = kummer_own_log(a,b,x)  # TODO: We cannot do negative x, the functions is in log
    else:
#        f_log = np.log(f_scipy)
        f_log = f_log
#        print (a,b,x)
#        print f_log
        
    f_log = float(f_log)
    return f_log


def kummer_own_log(a,b,x):
    # Default tolerance is tol = 1e-10.  Feel free to change this as needed.
    print ("$$$$$$$$$$$$$  Needed to use own Kummer func $$$$$$$$$$$$$$$$$$$$")
    tol = 1e-10;
    log_tol = np.log(tol)
    # Estimates the value by summing powers of the generalized hypergeometric
    # series:
    #      sum(n=0-->Inf)[(a)_n*x^n/{(b)_n*n!}
    # until the specified tolerance is acheived.
    
    log_term = np.log(x) + np.log(a) - np.log(b)
#    print a,b,x
#    f_log =  HMMl.sum_logs([0, log_term])
    
    n = 1;
    an = a;
    bn = b;
    nmin = 5;
    
    terms_list = []
    
    terms_list.extend([0,log_term])
    d = 0
    while((n < nmin) or (log_term > log_tol)):
      # We increase the n in 10 by 10 reduce overheading of  while
      n = n + d;
#      print "puto n %i"%(n)
#      print f_log
      an = an + d;
      bn = bn + d;
      
      d = 1
#      term = (x*term*an)/(bn*n);
      log_term1 = np.log(x) + log_term  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term2 = np.log(x) + log_term1  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term3 = np.log(x) + log_term2  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term4 = np.log(x) + log_term3  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
      d += 1
      log_term = np.log(x) + log_term4  + np.log(an+d) - np.log(bn+d) - np.log(n+d)
  
      terms_list.extend([log_term1,log_term2,log_term3,log_term4,log_term] )
      
      if(n > 10000):  # We fucked up
#        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4"
#        print " Not converged "
#        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4"
        # If we could not compute it, we raise an error...
        raise RuntimeError('Kummer function not converged after 10000 iterations. Kappa = %f', "Kummer_is_inf",x)
    f_log = gf.sum_logs(terms_list);
#    print "f_log success %f " % f_log
#    print "-----------------------------------------"
#    print n
#    print "-----------------------------------------"
    return f_log

def check_params(theta):
#    print "theta"
#    print len(theta)
#    print theta
    return check_Kummer(theta[0].size, theta[1])

def check_Kummer(Ndim, kappa):
    # This functions checks if the Kummer function will converge
    # Returns 1 if Kummer is stable, 0 if unstable
    a,b,x =  0.5, float(Ndim)/2, kappa
    try:
        kummer_log(a,b,x)
    except RuntimeError as err:
        return 0
    return 1