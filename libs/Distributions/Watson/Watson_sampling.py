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


def randWatsonMeanDir(N, k, p):

    min_thresh = 0 # 1/(float(100*N))

    step = 0.000001
    xx = np.arange(0, 1, step)

    yy = WatsonMeanDirDensity(xx, k, p)
    print (yy)
    cumyy = yy.cumsum(axis=0)*(xx[1]-xx[0])

    leftBound = xx[np.ndarray.flatten(np.asarray((np.nonzero(cumyy>min_thresh/2.0))))][0]

    xx = np.linspace(leftBound, 1, 1000)

    yy = WatsonMeanDirDensity(xx, k, p)

    M = yy.max()
    print (M)
    t = np.zeros((int(N),1))
    print (leftBound)
    for i in range (1, int(N)):
        while 1:
            x = np.random.uniform(0.0, 1.0)*(1-leftBound)+leftBound
            h = WatsonMeanDirDensity(x, k, p)
            draw = np.random.uniform(0.0, 1.0)*M
            if draw<=h:
                break
        if np.random.uniform(0.0, 1.0)>0.5:
            t[i] = x
        else:
            t[i] = -x

    return np.asarray(t)


def WatsonMeanDirDensity(x, k, p):
    Coeff = gamma(p/2.0) * (gamma((p - 1.0) / 2.0) * np.sqrt(np.pi) / hyp1f1(1.0/2.0, p/2.0, k))
    y = Coeff*np.exp(k*(np.power(x,2.0)))*np.power(1.0-x*x,(p-3.0)/2.0)
    return y

def randWatsonMeanDir2(N, kappa, p):
    # Generate samples of the unidimensional watson (0,..,0,1)
    # p is for the scalling of the cp
    ## For some reason this is multiplied by exactly p-1
    compute_leftBound =0
#    kappa = kappa/(p-1)
    print (p)
    normali = np.exp(Wad.get_cp_log(p,kappa) - Wad.get_cp_log(2,kappa)) ## TODO: It is a 2 right ?
    print ("Normalization %f" % normali)
    
    if (compute_leftBound):
        min_thresh = 1/(float(5)) #
        ### THIS IS JUST TO GET LEFT BOUNDARY AND PDF BOUNDATY ?
        step = 0.00001
        xx = np.arange(0, 1+step, step)
        xx = np.array(xx) * 2 * np.pi
        xx = np.array([np.cos(xx ), np.sin(xx)])
        
#        print xx.shape
#        xx = xx.T

        mux = np.array([1,0])
        
        # Get a grid of the univariate Watson
        yy = np.exp(Wad.Watson_pdf_log(xx, mux, kappa)) * normali

        # Get the cumulative distribution
        cumyy = yy.cumsum(axis=0)*(xx[1]-xx[0])
        
        print (np.max(yy))
#        print yy
        # Take care of the Boundaries
#        leftBound = xx[np.ndarray.flatten(np.asarray((np.nonzero(cumyy>min_thresh/2.0))))][0]
        leftBound = 0
#        print leftBound
    else:
        leftBound = 0.000
    leftBound = 0.000
#    print leftBound
    # Get the maximum probability of one of the samples
    if (kappa > 0):
        M =  np.exp(Wad.get_cp_log(p,kappa)) * np.exp(kappa)
        print ("Kappa positive, M: %f" % M)
    else:
        M = np.exp(Wad.get_cp_log(p,kappa)) * 1 #np.exp(-kappa)
        print ("Kappa negative, M: %f" % M)
#        print M
        
    t = np.zeros(int(N))
#    leftBound = 0.0
    # For every sample
    for i in range (0, int(N)):
        while 1:
            # TODO: Obtain a lot of samples first from np.random.uniform() snd
            # do all the process vectorialy.
        
            # Get uniform distribution in the limits
            x = np.random.uniform(0.0, 1.0)*(1-leftBound)+leftBound
#            x = [x, 0]
#            x = np.array(x) * 2 * np.pi
#            x = np.array([np.cos(x), np.sin(x)])
#            mux = np.array([1,0])
            # Compute the pdf of the random sample
#            h = np.exp(Wad.Watson_pdf_log(x, mux, kappa)) # * normali
            h = WatsonMeanDirDensity(x, kappa, p)

#            print h
            # If the sample pdf is bigger than the M we stop
            draw = np.random.uniform(0.0, 1.0)*M* (0.999999)
            
            ## TODO: Here is the shit to avoid the problem of that the maximum does not happen
            if draw <=h:
                break

        if np.random.uniform(0.0, 1.0)>0.5:
#            print 2
            t[i] = float(x)
        else:
            t[i] = -float(x)
    return np.asarray(t)

def randUniformSphere(N, p):
    # Generate N random vectors in the 1 sphere of dimension p
    randNorm = np.random.normal(0, 1, size=[N, p])
    RandSphere = np.zeros((N, p))

    for r in range(0, N):
        RandSphere[r,] = np.divide(randNorm[r,], np.linalg.norm(randNorm[r,]))
    return RandSphere

def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()

def randWatson(N, mu, k):
    # Generates N samples of a Watson distribution 
    # with mu and kappa
#    muShape = np.shape(mu)
#    print muShape
#    p = muShape[0]
    p =  np.array(mu).size
    tmpMu = np.zeros(p)
    tmpMu[0] = 1
    
    # 
    t = randWatsonMeanDir(N, k, p)
#    print t.shape
    RandSphere = randUniformSphere(int(N), p - 1)

    t_m = np.tile(t, (p, 1)).transpose()
    tmpMu_m = np.tile(tmpMu, (N, 1))

    t_m2 = np.tile(((1 - t**2)**(0.5)), [p, 1]).transpose()
    RNDS = np.c_[np.zeros(int(N)), np.asarray(RandSphere)]

    RandWatson = t_m * tmpMu_m + t_m2*RNDS

    # Rotate the distribution to the right direction
    Otho = null(np.matrix(mu))

    Rot = np.c_[mu, Otho]
    RandWatson = (Rot * RandWatson.transpose()).conj()
    
    return np.array(RandWatson).T


    