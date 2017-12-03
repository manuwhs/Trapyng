#########################################################3
############### BASIC MATH ##############################
##########################################################
## Library with basic mathematical functions 
# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
import matplotlib.colors as ColCon
from scipy import spatial
import datetime as dt
from sklearn import linear_model
import utilities_lib as ul

def get_return(price_sequences, lag = 1, cval = 0):
    # Get the return of the price sequences
    Nsam, Nsec = price_sequences.shape
#    print price_sequences.shape
    R = (price_sequences[lag:,:] - price_sequences[0:-lag,:])/price_sequences[0:-lag,:]
    # Add zero vector so that the length of the output is the same
    # as of the input
    cval_vec = np.ones((lag,Nsec)) * cval
    R = np.concatenate((cval_vec,R), axis = 0)
    return R

def reconstruc_return(seq,ret, lag = 1, cval = 0):
    # Reconstruction, given some returns a sequence
    # How it would be to take reconstruct from the diff signal

    # We shift to the right to that 
    # v[i] = x[i] - x[i-1] and x[i]
    # If we do v[j] = v[i+1] = x[i+1] - x[i]
    # Then v[j] + x[i] = x[i+1]
    # So then we shift to the left

    # For the returns is similar
    # r[i] = (x[i] - x[i-1])/x[i-1] and x[i]
    # x[i] = x[i-1]* r[i] + x[i-1]
    ret_shifted = np.roll(ret,-lag, axis = 0)
    reconstruction = seq * ret_shifted + seq
    reconstruction = np.roll(reconstruction,lag, axis = 0)
    return reconstruction

def get_cumReturn(price_sequences):
    # Get the cumulative return of the price sequences
    returns = get_return(price_sequences)
    cR = np.cumsum(returns, axis = 0)
    return cR
    
def get_SharpR(Returns, axis = 0, Rf = 0):
    # Computes the Sharp ratio of the given returns
    # Rf = Risk-free return

    E_Return = np.mean(Returns,axis)
    std_Return = np.std(Returns,axis)
    SR = (E_Return- Rf)/std_Return
    return SR

def get_SortinoR(Returns, axis = 0):
    # Computes the Sortino ratio of the given returns
    E_Return = np.mean(Returns,axis)
    Positive_Returns = Returns[np.where(Returns < 0)]
    std_Return = np.std(Positive_Returns,axis)
    SR = E_Return/std_Return
    return SR

def get_covMatrix(returns):
    # We need to transpose it to fit the numpy standard
    covRet = np.cov(returns.T)
    return covRet
    
def get_corrMatrix(returns):
    # We need to transpose it to fit the numpy standard
    covRet = np.corrcoef(returns.T)
    return covRet
    
def get_linearRef(X,Y):
    ## Calculates the parameters of the linear regression
    
    Nsam, Ndim = X.shape
#    X = np.concatenate((np.ones((Nsam,1)), X ), axis = 1)
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X, Y)

#    coeffs = np.array([regr.intercept_, regr.coef_])[0]
    
    coeffs = np.append(regr.intercept_, regr.coef_)
#    coeffs = np.concatenate((regr.coef_,regr.intercept_), axis = 1)
    return coeffs
    

def obtain_equation_line(Rf, Epoint, STDpoint):
    # Given the Rf and a portfolio point
    # This function calculates the equation of the line
    # Where the portfolio should be.
    P1 = [STDpoint, Epoint]
    P0 = [0, Rf]   # Origin point
    
    slope = (P1[1] - P0[1])/(P1[0] - P0[0])
    bias = slope * P0[0] + P0[1]  
    
    param = [bias, slope]
    
    return param
    
def get_TurnOver(w1,w2):
    # Gets the turn over between two porfolios
    # That is the absolute value of allocation we have to do.
    to = np.subtract(w1,w2)
    to = np.abs(to)
    to = np.sum(to)
    
    return to
    
def get_meanRange(timeSeries, window = 6):
    # This function moves a window of samples over a timeSeries and calculates
    # the mean and range to see if they are correlated and a transformation
    # of the original signal is needed.

    timeSeries = ul.fnp(timeSeries)
    means = []
    ranges = []
    
#    print timeSeries
    for i in range(timeSeries.size - window):   
        samples = timeSeries[i:(i+window),:]
        rangei = max(samples) - min(samples)
        meani = np.mean(samples)
        
        means.append(meani)
        ranges.append(rangei)
    
    means = ul.fnp(means)
    ranges = ul.fnp(ranges)

    return means, ranges
    
#from scipy.ndimage import interpolation
#interpolation.shift
    
def diff(X, lag = 1, n = 1, cval = 0): # cval=np.NaN
    # It computes the lagged difference of the signal X[Nsamples, Nsig]
    # The output vector has the same length as X, the noncomputable values
    # are set as cval
    # n is the number of times we apply the diff.
    X = ul.fnp(X)
    Nsa, Nsig = X.shape
    
    for i in range(n):
        X = X[lag:,:] - X[:-lag,:]
#        print sd
        # Shift to the right npossitions
    unk_vec = np.ones((lag*n,Nsig)) * cval
    X = np.concatenate((unk_vec,X), axis = 0)
    return X
    
def shift(X, lag = 1, cval = 0): # cval=np.NaN
    # It shifts the X[Nsam][Ndim] lag positions to the right. or left if negative
    X = ul.fnp(X)
    Nsa, Nsig = X.shape
    
    if (lag > 0):
        filling = cval * np.ones((lag,Nsig))
        X = np.concatenate((filling,X[:-lag,:]), axis = 0)
    elif(lag < 0):
#        print sd
        # Shift to the right npossitions
        filling = cval * np.ones((-lag,Nsig))
        X = np.concatenate((X[-lag:,:],filling), axis = 0)

    return X


def get_plane_Z(grid_x, grid_y, params):
    # Given the grids and the parameters we can calculate the Z
    xx, yy = np.meshgrid(grid_x, grid_y, sparse=True)
    z = params[0] + params[1]*xx  + params[2]*yy
    
    ## Method 2
#    z = []
#    for x in xx:
#        for y in yy:
#            z.append(params[0] + params[1]*x + params[2]*y)
#    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    return z
    