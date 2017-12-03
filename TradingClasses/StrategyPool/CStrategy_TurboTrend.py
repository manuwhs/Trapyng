
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utilities_lib as ul
import graph_lib as gr

import datetime as dt

def KNNPrice(self, L, K, algo, Npast = -1):
    # L: Length of the pattern
    # K: Number of most simmilar patterns
    # Npast: Number of past patterns to use.

    # Uses the Knn of the K most similar values for a pattern of length L
    # Simmilarity is equal 
    Prices = self.symbol.TDs[60].get_timeSeriesCumReturn()
    
    ## TODO -> Closser patterns or patterns in the same periodic state (check periodicity) hace preference
    ## Also, inside a sample, prices that are closer should have more influence        
    
    # TODO -> Incorporate metrics for Volume also. 
    # TODO -> Try doinf this with open, close, high, low or other shit instead
        
    X,Y = ul.windowSample(Prices,L)
    Nsamples, Ndim = X.shape

    Y = Y - X[:,-1].reshape((Nsamples,1))        # Y is the difference
    X[:] = X[:] - X[:,0].reshape((Nsamples,1))   # Center all

    
    ratioTr = 0.8   # Ratio of initial training samples to use
    startTst = int(ratioTr*Nsamples)   # Start point for tst

    print Prices.shape
    print startTst
    
    Nsucess = 0
    for i in range (startTst, Nsamples):  # Check the error for all of them
        msg = str(i) + " / " + str(Nsamples)
        print msg
        # Similarity between the last N patterns and the current patterns
        if (Npast == -1):
            pasado = i
        else:
            pasado = Npast
            
        sims_ored, sims_or = ul.simmilarity(X[i-pasado:i],X[i],algo)
        
#        print sims_ored
        # Get the BEST K indexes:
        BEST_K = sims_or[0:K]        
        
        if (i == 1996):
            # plot all the simmilar values 
            labels = ["PatterRecog1", "Time", "price"]
            gr.plot_graph([],X[BEST_K].T,labels, 1)
            
            # plot the value to estimate 
            labels = ["PatterRecog1", "Time", "price"]
            gr.plot_graph([],X[i],labels, 1)
        
        out = np.mean(Y[BEST_K])
        
        if (out*Y[i] > 0):
            Nsucess += 1
#            print out, Y[i]
    
    print "Success Rate: " + str(float(Nsucess)/(Nsamples-startTst))
    # We only use a training window of the las Tsamples ? 
    # We cannot use future patterns when seing the new patterns.
    # So the patterns we can use for identification are all the previous ones
    # X[0:startTst] are all the initial patterns we have.
    # As we try to get the tst for further points, we incorporate new patterns