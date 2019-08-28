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


def get_vonMissesFisher_muKappa_ML(X, rk = None, parameters = None):
    """Maxiumum likelihood estimator for a single vonMises"""
    
    Niter = parameters["Num_Newton_iterations"]
    try:
        # We compute the Weighted correlation matrix for the component
        # Sk = (N*pimix_k) * B_ik (*) X*X.T
        # If No rk specified it is just one 
        N,D = X.shape
        if(type(rk) == type(None)):
            rk = np.ones((N,1))*(1/float(N))
            ## TODO: better manage this
        if (1):
            # For the combination of Gaussian and Watson, we need to do the preprocessing here
            X = gf.remove_module (X)
        #################### Get the Mus from here !!
    
        N,p = X.shape
#        print X.shape, rk.shape
#        print(X*rk).shape
        sum_x = np.sum(X*rk,0)
        
#        print sum_x.shape
#        print rk
        sum_rk  = np.sum(rk)
        norm_sum_x = np.linalg.norm(sum_x)
        
        if (norm_sum_x < 0.00001):
            # Case where we have a degenerated cluster
    #        print "Degenerated cluster"
            raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', "Degenerated_Cluster_Error",np.sum(rk)/N)
            
            
        mu = sum_x/norm_sum_x
        R = norm_sum_x /(sum_rk)
        
#        print N,p,R
#
        tolerance =  1e-3
        
        if ((R > 1-tolerance) or (R < tolerance)):
            # Case where we have a degenerated cluster
    #        print "Degenerated cluster"
            raise RuntimeError('Degenerated cluster focus in one sample. Percentage_samples = %f', "Degenerated_Cluster_Error",np.sum(rk)/N)
            
            
        if (R > 0.9):    # When r -> 1 
            R = R - 1e-30  # To avoid divide by 0
            kappa0 = (R*(p-np.power(R,2)))/(1-np.power(R,2))
        elif(R < 0.1):    # When r -> 0
            R = R + 1e-30  # To avoid divide by 0
            kappa0 = (R*(p-np.power(R,2)))/(1-np.power(R,2))
        else:            # General approximation
            kappa0 = (R*(p-np.power(R,2)))/(1-np.power(R,2))


#        print "R: ", R
        
#        kappa0 = (R*p - np.power(R,3))/(1-np.power(R,2))
#        print "kappa: ", kappa0
#
        kappa_opt = Newton_kappa_log(kappa0,D,R,Niter)
#        
#        kappa1 = kappa0 - (A - R)/(1-np.power(A,2)-A*float(p-1)/kappa0)
        
#        kappa0 = np.min([1000, kappa0])
#        print "kappa Post: ", kappa_opt
        theta = [mu.reshape(D,1), kappa_opt.reshape(1,1)]

    except RuntimeError as err:
        theta = None
            
    return theta



## Own implementation of newton
def Newton_kappa_log(kappa0,D,R, Niter = None):
    kappa = kappa0
    D = float(D)
    
#    print "Init kappa: ", kappa, "Init R: " , R
    for i in range(Niter):
#        print kappa
        Ap_num = mpmath.besseli(D/2,float(kappa))
        Ap_den = mpmath.besseli(D/2-1,float(kappa))
        
        Ap = Ap_num/Ap_den
#        print Ap
        num = Ap - R
        den = 1 - Ap*Ap - ((D-1)/kappa)*Ap
#        print "delta_kappa:", num/den
        kappa = kappa - num/den
    
    return np.array(float(kappa))
