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
import scipy.stats as stats
from scipy.stats import multivariate_normal
import scipy

from sklearn.decomposition import PCA    
import math

def delta(L):
    delta = np.zeros((L,1))
    delta[0] = 1
    return delta

def diffw(L, lag = 1): # Window for the lagging
    delta = np.zeros((L,1))
    delta[0] = 1
    delta[lag] = -1
    return delta

def convolve(signal, window, mode = "full"):
    # performs the convolution of the signals
    L = window.size
    sM = np.convolve(signal.flatten(),window.flatten(), mode = "full")
    
    if (mode == "valid"):
        sM[:L] = sM[:L] * np.Nan
        sM = sM[:-L+1]    # Remove the last values since they hare convolved with 0's as well
    sM = sM.reshape ((sM.size,1))
    return sM

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
    
def diff(X, lag = 1, n = 1, cval = np.nan): # cval=np.NaN
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

#theta = np.degrees(np.arctan2(*vecs[:,0][::-1])) # *vecs[:,0][::-1]
#def length(v):
#  return np.sqrt(v.T.dot(v))
#theta = np.degrees(math.acos(ret1.T.dot(ret2) / (length(ret1) * length(ret2))))

def eigsorted2(data):
    pca = PCA(n_components=2)
    xtPCA = pca.fit_transform(data)
    vecs = pca.components_ 
    vals = pca.explained_variance_
    return vecs,vals

def gaussian1D_points(X = None, mean = None, std = None, 
                      num = 100, std_K = 2, x_grid = None):
    ## Fit a gaussian to the data or use the parameters given.
    # Gives the gaussian set of points for the std_K
    if (type(X) != type(None)):
        mean = np.mean(X)
        std = np.std(X)
    
    if (type(x_grid) == type(None)):
        x_grid = np.linspace(mean - std_K*std, mean + std_K*std, num = num)
        
    y_values = multivariate_normal.pdf(x_grid,mean,std*std)
#    Z = (x_grid - mean)/std
#    y_values = stats.norm.pdf(Z) / std# * stats.norm.pdf(-mean/std)

    return x_grid, y_values

def gaussian1D_points_cdf(X = None, mean = None, std = None, 
                      num = 100, std_K = 2, x_grid = None):
    ## Fit a gaussian to the data or use the parameters given.
    # Gives the cummulative distribution of the gaussian set of points for the std_K
    if (type(X) != type(None)):
        mean = np.mean(X)
        std = np.std(X)
    
    if (type(x_grid) == type(None)):
        x_grid = np.linspace(mean - std_K*std, mean + std_K*std, num = num)
    
    # We normalize since the function assumes a gaussian N(0,1)    
    Z = (x_grid - mean)/std
    y_values = stats.norm.cdf(Z) # * stats.norm.pdf(-mean/std)

    return x_grid, y_values

def empirical_1D_cdf(X):
    ## Fit a gaussian to the data or use the parameters given.
    # Gives the gaussian set of points for the std_K
    
    sorted_X = np.sort(X.flatten())
    y_values = ul.fnp(range(1,X.size + 1))/float(X.size)

    return sorted_X, y_values
    
def get_eigenVectorsAndValues(X = None, Sigma = None):
    # Gets the gaussian params needed to plot the ellipse of data
    if (type(X) == type(None)):
        vecs, vals, V = scipy.linalg.svd(Sigma)
    else:
        vecs,vals = eigsorted2(X)
    return vecs,vals
    
def get_gaussian_ellipse_params(X = None, mu = None, Sigma = None, Chi2val = 2.4477):
    # Gets the gaussian params needed to plot the ellipse of data
    
    if (type(X) != type(None)):
        mean =  np.mean(X, axis = 0)
    #    vals, vecs = eigsorted(cov)
        vecs,vals = eigsorted2(X)
    
    else:
         vecs, vals, V = scipy.linalg.svd(Sigma)
         mean = mu
    # Get the 95% confidence interval error ellipse
    w, h = Chi2val * np.sqrt(vals)
    # Calculate the angle between the x-axis and the largest eigenvector
    
    # This angle is between -pi and pi.
    #Let's shift it such that the angle is between 0 and 2pi
    # theta
    theta = math.atan2(vecs[0,1],vecs[0,0]);
    if(theta < 0):
        theta = theta + 2*np.pi;
    return mean, w,h, theta
    
def get_ellipse_points(center,a,b,phi, num = 100):
    # Returns: the [x,y]_Nsam points to plot an ellipse.
    # center: Centre point of the ellipse
    # a: X width of the ellipse
    # b: Y height of the llipse
    # phi: Angle of the major axis of the ellipse with the X axis.

    theta_grid = np.linspace(0,2*np.pi, num = num);
    
    # the ellipse in x and y coordinates 
    ellipse_x_r  = a*np.cos( theta_grid );
    ellipse_y_r  = b*np.sin( theta_grid );
    #Define a rotation matrix
    R = np.array([[ np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi) ]]);
    #let's rotate the ellipse to some angle phi
    r_ellipse = np.array([ellipse_x_r,ellipse_y_r]).T.dot(R)
    r_ellipse[:,0] += center[0]
    r_ellipse[:,1] += center[1]
    return r_ellipse

from sklearn.neighbors import KernelDensity
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    
    x = ul.fnp(x)
    x_grid = ul.fnp(x_grid)
    print x.shape
#    if (x.shape[1] == 1):
#        x = x[:, np.newaxis]
#        x_grid = x_grid[:, np.newaxis]
        
    kde_skl.fit(x)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid)
    return np.exp(log_pdf)

def kde2D(x, y, bandwidth, xbins=10j, ybins=10j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y.T, x.T]).T
    
    print xy_train.shape
    print xy_sample.shape
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

def get_gaussian2D_pdf(data = None, xbins=10j, ybins=10j, mu = None, cov = None, 
                      std_K = 2, x_grid = None):
    ## Fit a gaussian to the data or use the parameters given.
    # Gives the gaussian set of points for the std_K
    mu = np.array(mu).flatten()
    std_1 = np.sqrt(cov[0,0])
    std_2 = np.sqrt(cov[1,1])
    if (type(data) != type(None)):
        mu = np.mean(data)
        cov = np.cpv(data)
    
    if (type(x_grid) == type(None)):
        xx, yy = np.mgrid[mu[0] - std_K*std_1:mu[0] + std_K*std_1:xbins, 
                          mu[1] - std_K*std_2:mu[1] + std_K*std_2:ybins]

    # Function to obtain the 3D plot of the pdf of a 2D gaussian
    # create grid of sample locations (default: 100x100)

    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
#    xy_train  = np.vstack([data[:,[1]].T, data[:,[0]].T]).T
    

    # score_samples() returns the log-likelihood of the samples
    z = multivariate_normal.pdf(xy_sample,mu,cov)
    return xx, yy, np.reshape(z, xx.shape)

