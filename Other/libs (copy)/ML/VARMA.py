
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

from statsmodels.tsa import stattools
import statsmodels.tsa.seasonal as tsa_seasonal
from statsmodels.tsa import arima_model
##########################################################
############### BASIC TIME SERIES ANALYSIS ###############
##########################################################

# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]

def acf(timeSeries, nlags = 40, alpha = 0.05, unbiased = True, qstat = False):
    # Get the return of the price sequences
    results = stattools.acf( timeSeries, nlags = nlags, 
                      alpha = alpha, unbiased = unbiased, 
                      qstat = qstat)
    return results

def pacf(timeSeries,nlags = 40, alpha = 0.05,   method = 'ywunbiased'):
    results  = stattools.pacf(timeSeries, nlags = nlags, 
                            alpha = alpha, 
                            method = method)

    return results

def seasonal_decompose(timeSeries, freq = 34):
    # Seasonal decomposition using moving averages
    decomposition = tsa_seasonal.seasonal_decompose(timeSeries, freq = freq)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return [trend, seasonal, residual]
    
def ccf(ts1, ts2, unbiased = True):
    ## cross-correlation function for 1d
    values_ccf = stattools.ccf(ts1, ts2, unbiased)	

def periodogram(timeSeries):
    # Returns the periodogram for the natural frequency of X
    values_periodograk = stattools.periodogram(timeSeries)	


def ARIMA(timeSeries, order=(2, 1, 2)):
    model = ARIMA(timeSeries, order=(2, 1, 2))  
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))