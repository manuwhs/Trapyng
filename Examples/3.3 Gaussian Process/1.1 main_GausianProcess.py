""" SIGNAL ANALISIS AND PREPROCESSING:
  - ARMA
  - Fourier and Wavelet
  - Filters: Kalman, Gaussian process 
  """
import os
os.chdir("../../")
import import_folders

# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
import pylab
# Own graphical library
from graph_lib import gl
import graph_tsa as grtsa
# Data Structures Data
import CTimeData as CTD
# Import functions independent of DataStructure
import utilities_lib as ul
import indicators_lib as indl
import indicators_pandas as indp
import oscillators_lib as oscl

# Import specific model libraries
import GaussianProcess as GPown
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared
from sklearn import preprocessing

plt.close("all") # Close all previous Windows
######## SELECT DATASET, SYMBOLS AND PERIODS ########
dataSource =  "GCI"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)

symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
symbols = ["Amazon"]
periods = [1440]
######## SELECT DATE LIMITS ###########
sdate_str = "01-01-2016"
edate_str = "2-1-2017"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Tell which company and which period we want
timeData = CTD.CTimeData(symbols[0],periods[0])
timeData.set_csv(storage_folder)  # Load the data into the model
timeData.set_interval(sdate,edate) # Set the interval period to be analysed


GP_toyExample = 0
GP_sklearn = 0
GP_own = 1


if (GP_toyExample):
    np.random.seed(1)
    def f(x):
        return x * np.sin(x)

    # ----------------------------------------------------------------------
    X = np.linspace(0.1, 9.9, 20)
    X = np.atleast_2d(X).T
    
    # Observations and noise
    y = f(X).ravel()
    # dy is the noise variance std for each sample
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    
    # Mesh the input space for evaluations of the real function, the prediction and
    x = np.atleast_2d(np.linspace(0, 15, 1000)).T
    
    # Instanciate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    + WhiteKernel(1e-1)
    
    # Instanciate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(dy / y) ** 2,
                                  n_restarts_optimizer=10)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    
    gl.plot_timeRegression(x, y_pred, sigma,
                            X, y,dy, 
                            labels = ["GP sklearn toy","Time","Price"])
    gl.plot(x, f(x), legend=[u'$Real f(x) = x\,\sin(x)$'],  nf = 0)

#################################################################

if ( GP_sklearn):
    timeSeries = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()
    #%% Normalize data
    scaler = preprocessing.StandardScaler().fit(timeSeries)
    timeSeries = scaler.transform(timeSeries) 
    #    timeSeries = scaler.transform(timeSeries)    
    dates = ul.fnp(ul.datesToNumbers(dates))
    x_grid = np.atleast_2d(np.linspace(dates[0], dates[-1], 1000)).T
    
    X = ul.fnp(np.array(dates))
    y = timeSeries.reshape(timeSeries.size,1)
    
    # Define the kernel and the GP object.
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3))
    sigma_eps = 0.1
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10,
                                  alpha= sigma_eps)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x_grid, return_std=True)
    sigma = ul.fnp(sigma)
    # Plot the function, the prediction and the 95% confidence interval based on

    gl.plot_timeRegression(x_grid, y_pred, sigma,
                            dates, timeSeries,sigma_eps, 
                            labels = ["GP sklearn","Time","Price"])
    
############## TIME SERIES INDICATORS #####################################

if (GP_own):
        #%% Normalize data

    
    ## Get the data and transform it to be a normal ML problem data
    timeSeries = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()
    scaler = preprocessing.StandardScaler().fit(timeSeries)
    timeSeries = scaler.fit_transform(timeSeries)    
    dates = ul.fnp(ul.datesToNumbers(dates))
    
    dates = ul.fnp(range(dates.size))
    ## Generate a Validation set and obtain the regressed values !
    x_grid = np.atleast_2d(np.linspace(dates[0], dates[-1] + (dates[-1]- dates[0])*0.1, 100)).T
    
    ### Initial Parameters of the problem ####
    l = 0.01
    sigma_0 = 0.9
    sigma_eps = 0.5
    
    ## Create an instance of the gp, train it
    gp = GPown.GaussianProcessRegressor()
    
    gp.fit(dates,timeSeries, 
           kernel = GPown.compute_Kernel, 
           params = dict([["l",l],["sigma_0",sigma_0]]),
            sigma_eps = sigma_eps)
    

    y_pred, Cov = gp.predict(x_grid)
    sigma = ul.fnp(np.sqrt(np.diag(Cov)))

    ## Plot the results
    gl.plot_timeRegression(x_grid, y_pred, sigma,
                            dates, timeSeries,sigma_eps, 
                            labels = ["GP own implementation","Time","Price"], nf = 1)
    
    optflag = 1
    if (optflag):
        ### Optimize the parameters and do it again ####
        xopt = gp.optimize_parameters(sigma_0, l, sigma_eps)
        sigma_0, l, sigma_eps = xopt
        gp.fit(dates,timeSeries, 
               kernel = GPown.compute_Kernel, 
               params = dict([["l",l],["sigma_0",sigma_0]]),
                sigma_eps = sigma_eps)
        ## Generate a Validation set and obtain the regressed values !
        y_pred, Cov = gp.predict(x_grid)
        sigma = ul.fnp(np.sqrt(np.diag(Cov)))
        gl.plot_timeRegression(x_grid, y_pred, sigma,
                                dates, timeSeries,sigma_eps, 
                                labels = ["GP Estimation (Opt)","Time","Price"], nf = 1)
                    
        ## Plot realizations !! 
    #    f_s = gp.generate_process(x_grid, N = 20)
    #    gl.plot(x_grid,f_s, legend = ["Realization"], nf = 0)
    timeSeriesflag = 1
    if (timeSeriesflag):
        # TODO: The peaks of variance are cause by the weekends ! They are further.
        # How to model this in the algorithm !
        sigma_eps = None # For not plotting shit
        
        ws = 20
        lag = 1
        y_pred, Cov = gp.OneStepWindowedPrediction(dates,timeSeries, ws = ws,lag = lag)
        y_pred = np.array(y_pred)
        Cov = np.array(Cov)
        sigma = np.sqrt(Cov)

        gl.plot_timeRegression(dates[ws+lag:y_pred.size + ws+lag,:], y_pred, sigma,
                                dates, timeSeries,sigma_eps, 
                                labels = ["GP OneStepPrediction lag = %i (Opt)"%lag,"Time","Price"], nf = 1)
    
        ws = 20
        lag = -4     
        y_pred, Cov = gp.OneStepWindowedPrediction(dates,timeSeries, ws = ws, lag = lag)
        y_pred = np.array(y_pred)
        Cov = np.array(Cov)
        sigma = np.sqrt(Cov)
        gl.plot_timeRegression(dates[ws+lag:y_pred.size + ws+lag,:], y_pred, sigma,
                                dates, timeSeries,sigma_eps, 
                               labels = ["GP OneStepPrediction lag = %i (Opt)"%lag,"Time","Price"], nf = 1)
   # El futuro da info del pasado.
   # Senales mas hijas de puta crearan mas fallos en la prediccion one step.
   # Calcular para distintas senales cuanto fallamos en la prediccion.
                            