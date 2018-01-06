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
import HMM_libfunc2 as HMMl

import Watson_distribution as Wad
import Watson_sampling as Was
import Watson_estimators as Wae
import general_func as gf

def get_MLmean(X, S = None):
    n,d = X.shape
    # Check if we are given the S
    # Maybe not used in the end, it has to be done together with Kappa
    # To check if we need to return mu1 or mup
    if (type(S) == type(None)):
        S = np.dot(X.T,X)   # Correlation
    S = S/n             # Not really necesarry
    D,V = np.linalg.eig(S) # Obtain eigenvalues D and vectors V
    
    if (D[0] == D[1]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    return V[:,0]

def get_Weighted_MLMean(rk,X, Sk = None):
    
    n,d = X.shape
    # Check if we are given the S
    if (type(Sk) == type(None)):
        Sk = np.dot(X.T,rk*X)   # Correlation # We weight the samples by the cluster responsabilities r[:,k]
        # Correlation
    
    Sk = Sk/n             # Not really necesarry
    D,V = np.linalg.eig(Sk) # Obtain eigenvalues D and vectors V
    
    max_d = np.argmax(D)
    if (D[max_d] == D[max_d]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    return V[:,max_d]

def get_kappaNewton(k, args):  # The r is computed outsite
    Ndim = args[0]
    r = args[1]
    
    a = 0.5
    c = float(Ndim)/2
    
    M = np.exp(Wad.kummer_log(a, c,k))
    Mplus = np.exp(Wad.kummer_log(a + 1, c +1,k))
    dM = (a/c)*Mplus 

    g = dM/M
#    kummer = 
#    print Ndim, k, r
    return g - r

def Newton_kappa(kappa0,Ndim,r, Ninter = 10):
    kappa = kappa0
    a = 0.5
    c = float(Ndim)/2
    for i in range(Ninter):
        
        M = np.exp(kummer_log(a, c,kappa))
        Mplus = np.exp(kummer_log(a + 1, c +1,kappa))
        dM = (a/c)*Mplus 
#        dM = (a - c)*Mbplus/c + M
        g = dM/M
#        print g
        dg =  (1 - c/kappa)*g + (a/kappa) - g*g
        
        kappa = kappa - (g - r)/dg
        
#        print kappa
    return kappa
    
def get_MLkappa(mu,X, S = None):
    n,d = X.shape
    if (type(S) == type(None)):
        S = np.dot(X.T,X)   # Correlation
        
    S = S/n
    r = np.dot(mu.T,S).dot(mu)
#    print r
    
    a = 0.5
    c = float(d)/2
    
    # General aproximation
#    BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    
    # When r -> 1 
    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
#    print BGG
    
#    BGG = Newton_kappa(BGG,d,r,Ninter = 5)
#    BGG = newton(get_kappaNewton, BGG, args=([d,r],))
#    print "STSHNWSRTNSRTNWRSTN"
    return BGG

def get_Watson_muKappa_ML(X):
    # This function obtains both efficiently and checking the sign and that
    n,d = X.shape
    a = 0.5
    c = float(d)/2
    
    S = np.dot(X.T,X)   # Correlation
    S = S/n             # Not really necesarry

    # Get eigenvalues to obtain the mu
    D,V = np.linalg.eig(S) # Obtain eigenvalues D and vectors V
    
    print D
    
    d_pos = np.argmax(D)
    d_min = np.argmin(D)
    print d_pos, d_min
    ## We first assume it is positive if not we change the mu
    if (D[0] == D[1]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    if (D[-1] == D[-2]):
        print "Warning: Eigenvaluep = EigenValuep-1 in MLmean estimation"

    ## We solve the positive and the negative situations and output the one with
    ## the highest likelihood ? 

    mu_pos = V[:,d_pos]  # This is the vector with the highest lambda
    mu_neg = V[:,d_min]  # This is the vector with the lowest lambda
    
    r_pos = np.dot(mu_pos.T,S).dot(mu_pos)
    r_neg = np.dot(mu_neg.T,S).dot(mu_neg)
#    print r

    # General aproximation
    BGG_pos = (c*r_pos -a)/(r_pos*(1-r_pos)) + r_pos/(2*c*(1-r_pos))
#    kappa_pos = BGG_pos
    kappa_pos = newton(get_kappaNewton, BGG_pos, args=([d,r_pos],))

    BGG_neg = (c*r_neg -a)/(r_neg*(1-r_neg)) + r_neg/(2*c*(1-r_neg))
#    kappa_neg = BGG_neg
    kappa_neg = newton(get_kappaNewton, BGG_neg, args=([d,r_neg],))
    
    likelihood_pos = np.sum(Wad.Watson_pdf_log(X.T,mu_pos,kappa_pos))
    likelihood_neg = np.sum(Wad.Watson_pdf_log(X.T,mu_neg,kappa_neg))
    
    print likelihood_pos, likelihood_neg
    print kappa_pos, kappa_neg
    if (likelihood_pos >=likelihood_neg):
        kappa = kappa_pos
        mu = mu_pos
    else:
        kappa = kappa_neg
        mu = mu_neg
    return mu, kappa

def get_Watson_Wighted_muKappa_ML(X, rk):
    # This function obtains both efficiently and checking the sign and that
    n,d = X.shape
    a = 0.5
    c = float(d)/2
#    print (X*rk).shape
    
    Sk = np.dot(X.T,X*rk)   # Correlation
    Sk = Sk/np.sum(rk)            # Not really necesarry

    # Get eigenvalues to obtain the mu
    D,V = np.linalg.eig(Sk) # Obtain eigenvalues D and vectors V
    
#    print D
    
    d_pos = np.argmax(D)
    d_min = np.argmin(D)
    ## We first assume it is positive if not we change the mu
    if (D[0] == D[1]):
        print "Warning: Eigenvalue1 = EigenValue2 in MLmean estimation"
    if (D[-1] == D[-2]):
        print "Warning: Eigenvaluep = EigenValuep-1 in MLmean estimation"

    ## We solve the positive and the negative situations and output the one with
    ## the highest likelihood ? 

    mu_pos = V[:,d_pos]  # This is the vector with the highest lambda
    mu_neg = V[:,d_min]  # This is the vector with the lowest lambda
    
    r_pos = np.dot(mu_pos.T,Sk).dot(mu_pos)
    r_neg = np.dot(mu_neg.T,Sk).dot(mu_neg)
#    print r

    # General aproximation
    BGG_pos = (c*r_pos -a)/(r_pos*(1-r_pos)) + r_pos/(2*c*(1-r_pos))
#    kappa_pos = BGG_pos
    kappa_pos = newton(get_kappaNewton, BGG_pos, args=([d,r_pos],))

    BGG_neg = (c*r_neg -a)/(r_neg*(1-r_neg)) + r_neg/(2*c*(1-r_neg))
#    kappa_neg = BGG_neg
    kappa_neg = newton(get_kappaNewton, BGG_neg, args=([d,r_neg],))
    
#    likelihood_pos = np.sum(Watson_pdf_log(X.T,mu_pos,kappa_pos))
#    likelihood_neg = np.sum(Watson_pdf_log(X.T,mu_neg,kappa_neg))

#    likelihood_pos = np.sum(np.exp(Watson_pdf_log(X.T,mu_pos,kappa_pos))*rk.T)
#    likelihood_neg = np.sum(np.exp(Watson_pdf_log(X.T,mu_neg,kappa_neg))*rk.T)

    # The maximum weighted likelihood estimator
    likelihood_pos = np.sum(Wad.Watson_pdf_log(X.T,mu_pos,kappa_pos)*rk.T)
    likelihood_neg = np.sum(Wad.Watson_pdf_log(X.T,mu_neg,kappa_neg)*rk.T)

 
    print likelihood_pos, likelihood_neg
    if (likelihood_pos > likelihood_neg):
        kappa = kappa_pos
        mu = mu_pos
    else:
        kappa = kappa_neg
        mu = mu_neg
    return mu, kappa
    
def get_Weighted_MLkappa(rk, mu,X, Sk = None):
    n,d = X.shape

    if (type(Sk) == type(None)):
        Sk = np.dot(X.T,rk*X)   # Correlation # We weight the samples by the cluster responsabilities r[:,k]
        # Correlation
#    print r
    Sk = Sk/(np.sum(rk))
    
    r = np.dot(mu.T,Sk).dot(mu)
    a = 0.5
    c = float(d)/2
    
    # General aproximation
    BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    
    # When r -> 1 
#    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
    # TODO: In some examples this does not converge in the scipy method..
#    BGG = newton(get_kappaNewton, BGG, args=([d,r],))
    BGG = Newton_kappa(BGG,d,r,Ninter = 30)
    return BGG

    