
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
import VARMA as tsa
from statsmodels.tsa import stattools
##########################################################
############### BASIC TIME SERIES ANALYSIS ###############
##########################################################

# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]

def plot_acf_pacf(timeSeries, nlags = 40, alpha = 0.05,
                  method_acf = True, method_pacf = 'ywunbiased',
                  legend = ["ACF", "PACF"],
                  labels = ["timeSeries"]):
                      
    valuesACF, confIntACF = tsa.acf(timeSeries, nlags = 30, 
                          alpha = 0.05, unbiased = True, 
                          qstat = False)
                          # qstat: For LJung-Box values and pvalues: For Pvalues of this
    ## For some reason the first value of the PACF is 1, when it should not be defined
    valuesPACF, confIntPACF  = tsa.pacf(timeSeries, nlags = 30, 
                            alpha = 0.05, 
                            method = 'ywunbiased')
                            
    gl.set_subplots(2,1)
    
    gl.stem([],valuesACF,
            labels = [labels[0], "lag", "ACF"],
            legend = [legend[0]])
            
#    gl.plot([],confIntACF, nf = 0, color = "blue",
#            legend = ["ConfInt %.2f" % (1 - alpha)],
#            lw = 1)
    
    plt.axhline(y=-1.96/np.sqrt(len(timeSeries)),
                linestyle='--',color='gray', lw = 3)
    plt.axhline(y=1.96/np.sqrt(len(timeSeries)),
                linestyle='--',color='gray',  lw = 3)

    gl.stem([],valuesPACF,
            labels = [labels[0], "lag", "PACF"],
            legend = [legend[1]])
            
#    gl.plot([],confIntPACF, nf = 0, color = "blue",
#            legend = ["ConfInt %.2f" % (1 - alpha)])
        
    plt.axhline(y=-1.96/np.sqrt(len(timeSeries)),
                linestyle='--',color='gray', lw = 3)
    plt.axhline(y=1.96/np.sqrt(len(timeSeries)),
                linestyle='--',color='gray',  lw = 3)
                
def plot_decomposition(timeSeries):
    trend, seasonal, residual = tsa.seasonal_decompose(timeSeries)
    
    gl.set_subplots(4,1)
    gl.plot([], timeSeries, 
            labels = ["", "time", "Original"],
            legend = ['Original'],
            loc = "best")
    
    gl.plot([], trend, 
            labels = ["", "time", "trend"],
            legend = ['trend'],
            loc = "best")
            
    gl.plot([], seasonal, 
            labels = ["", "time", "seasonal"],
            legend = ['seasonal'],
            loc = "best")
            
    gl.plot([], residual, 
            labels = ["", "time", "residual"],
            legend = ['residual'],
            loc = "best")
