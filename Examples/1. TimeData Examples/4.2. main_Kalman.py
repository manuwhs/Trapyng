""" SIGNAL ANALISIS AND PREPROCESSING:
  - ARMA
  - Fourier and Wavelet
  - Filters: Kalman, Gaussian process 
  """
import os
os.chdir("../")
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
import basicMathlib as bMA
import utilities_lib as ul
import indicators_lib as indl
import indicators_pandas as indp
import oscillators_lib as oscl
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

KM = 1

def OneStepPredLL(y, ypred, SigmaYYpredList):
    # Returns the log-likelihood of the likelihood of the prediction given
    # the model and the current state P(Ypred | Ypred)
    logpred = 0
    Ns,Ndim = y.shape
    
    yerr = ypred - y  # Difference at each point, between the predicted and the real.
    for i in range (Ns):
        yerr_i = yerr[[i],:]
        logpred += np.log(np.linalg.det(SigmaYYpredList[i]))
        logpred += (yerr_i.T).dot(np.linalg.inv(SigmaYYpredList[i])).dot(yerr_i)
        
    logpred = -logpred/2
    logpred -= Ns/np.log(np.pi * 2)
    return logpred


if (KM == 1):
    # Kalman Filter !
    timeSeries = timeData.get_timeSeries(["Close"]);
    returns = timeData.get_timeSeriesReturn()  # Variable to estimate. Nsig x Ndim
    returns  = bMA.diff(timeSeries)
    
    dates = timeData.get_dates()
    dates = ul.fnp(ul.datesToNumbers(dates))
    dates = ul.fnp(range(dates.size))
        
    ############################ KALMAN FILTER !! #######################################

    Matrix_tr = np.concatenate([timeSeries,returns], axis = 1)  # Nsam * 4
    
    Ns,Nd = Matrix_tr.shape
    Nout = 1 # Number of output
    # System of equations A
    #   Xt = A*Xt-1 + B*Ut + SystemNoise  
    #   Yt = Cx_k  + MeasurementNoise
    # In the normal formulation Xt is a column vector !!
    
    Matrix_tr = Matrix_tr.T
    ########################### AR matrix ###############################
    # We asume that x and y coordinates are independent 
    # and that the dependence between variables is:
    # P_t = P_(t-1) + vP_(t-1)  -> The dinamics of the next price
    # vxt = vx_(t-1)            -> The dinamics of the next return
    # This leads to the next Autocorrelation Matrix A
    A = np.array([[1,0.99],
                  [0,1]])
    
    ######################### C Matrix ###################################
    # Measurement matrix.
    # What transformation of the real state we perfrom.
    # In this case we just select both variables indpendently 
    C = np.eye(Nd)    # np.array([[1,1],[0,1]])
    
    # Dynamic System error Covariance Matrix
    # The error that ocurrs in the system due to variables that we are not taking

    ######################### B Matrix ###################################
    # In our case B*Ut is 0, we do not have acceleration modeled.
    
    
    ################# Noise Paramters. Covariance Matrixes ###################  
    # Parameters
    varPrice = 5
    varDiffPrice = 2# Because it is the substraction of 2 random variables of variance 1
    varNoisePrice =  20 # Variance of the observation noise.
    varNoiseDiff = 10  # Variance of the observation noise.
    
    # Covariance noise of the System
    SystemCovNoise = np.eye(Nd)   # Independent states
    SystemCovNoise[0,0] = varPrice
    SystemCovNoise[1,1] = varDiffPrice
    
    # Measurement error Covariance Matrix
    # The covariance matrix of the measurement error observed at a given time t.
    # We want to compute both price and return, so we obtain both
    # therefore the noise measurement matrix is a 2x2 covariance matrix
    MeasurCovNoise = np.eye(Nd) # Independent noise. This is not really true I think though. Can we check it ?
    MeasurCovNoise[0,0] = varNoisePrice
    MeasurCovNoise[1,1] = varNoiseDiff
    ####################### Initialize Kalman Filter ##########################
    ## Time to choose our priors of prediction !!
    ## Important: 
    #      - Variables with "hat" are estimated variables (V(t|t))
    #      - Variables with "pred" are predicted variables (V(t+1|t))
    # In the begining we have to define Xpred(1|0).
    # Observed samples start at time t = 1. t = 0 is our prior thingy.
    
    ## We just initialize the parameters of the kalman it with a good guess. 
    # In this case we just use the initial obervation of Yo as mean of the prior. X0hat = Y0
    Xpred = Matrix_tr[:,[0]] # Initilization Xpred for X1 given Y0
    
    # Initilize the uncertainty of the first samples high enough if we are not sure
    # or low enough if we are sure of out initial Xhat
    # As more samples arise, this influence of this initial decision will be decrease.
    # We choose a small uncertainty of the initializatoin
    varpred0 = 20
    SigmaXXpred  = np.eye(Nd) * varpred0
    ## Initial uncertainty of the observed variables. We say, small.
    ## Calculate covariance matrix between observarionts and latent variables
    SigmaXYpred  = SigmaXXpred.dot(C.T)
    # Maybe same as the measurement noise would be appropiate.
    SigmaYYpred = C.dot(SigmaXYpred) + MeasurCovNoise

    
    ######### Initialize variables just to keep intermediate results ##########
    XhatList = []               # Store here the estimations of X
    SigmaXXhatList = []        # Store here the estimated Sigma of X
    
    XpredList = []              # Store here the predictions of X
    SigmaXXpredList = []        # Store here the predicted Sigma of X
    
    
    XpredList.append(Xpred)           # Store here the predictions of X
    SigmaXXpredList.append(SigmaXXpred)       # Store here the predicted Sigma of X
    
    ###########################################################################
    ################## RECONSTRUCTION OF THE STATE ############################
    ###########################################################################
    for n in range (0,Ns):
      #############   ESTIMATING STEP  ###################
      # Estimation of theparameters ("hat" variables V(t|t))
      # Calculated from the predicted values and the new sample
      K = SigmaXYpred.dot(np.linalg.inv(SigmaYYpred))  # Kalman Gain
#      print  K.dot(C.dot(Matrix_tr[n,:]- Xpred))
      XpredError = Matrix_tr[:,[n]]- Xpred
#      print K
#      if (n == 5):
#         print XpredError
#         print  Matrix_tr[:,[n]]
#         print Xpred
#         
      Xhat = Xpred + K.dot(C.dot(XpredError))
      SigmaXXhat = SigmaXXpred - K.dot(SigmaYYpred.dot(K.T))
      
#      print K.shape, Xhat.shape, SigmaXXhat.shape
      #############   PREDICTION STEP  ###################
      ## Predict the variables for the next instance  V(t+1|t)
      Xpred = A.dot(Xhat)  # We use the correlation bit
      SigmaXXpred = A.dot(SigmaXXhat.dot(A.T)) + SystemCovNoise
      SigmaXYpred = SigmaXXpred.dot(C.T)
      SigmaYYpred = C.dot(SigmaXYpred) + MeasurCovNoise

#      print SigmaXXpred, SigmaXXhat.dot(A.T), A
      ################# Storing data  ################
      XhatList.append(Xhat)
      SigmaXXhatList.append(SigmaXXhat)
      XpredList.append(Xpred)
      SigmaXXpredList.append(SigmaXXpred)
  
    ######################################################################
    ############ Compare the Y estimated with the measurements !!! #######
    ######################################################################
    XhatList = np.concatenate(XhatList, axis = 1)
    XpredList = np.concatenate(XpredList, axis = 1)
    
    Yhat =  C.dot(XhatList)
    Ypred = C.dot( XpredList)
    diff = Yhat - Matrix_tr
    
    ##################  Predictions  #########################################v
    # Now, using the final state of training (last estimations), we will predict 
    # the state and observations for the last 50 samples (test samples)
    ## The initial hat is the one predicted for the last training sample
    # We initilize with last parameters calculated
        

    Xpredtest = Ypred[:,[-1]]
    SigmaXXpredtest = SigmaXXpredList[-1]
    
    ## Variables to store the results
    XpredtestList = []
    SigmaXXpredtestList = []
    
    # The first prediction is the one calculated last in the previous loop
    XpredtestList.append( Xpredtest)
    SigmaXXpredtestList.append(SigmaXXpredtest)
    
    Ntest = 10
    # We calculate the rest
    for n in range(Ntest):
      # Perform future predictions
      Xpredtest = A.dot(Xpredtest)
      SigmaXXpredtest = A.dot(SigmaXXpredtest.dot(A.T)) + SystemCovNoise 
      
      XpredtestList.append(Xpredtest)
      SigmaXXpredtestList.append(SigmaXXpredtest)
    
    
    ##################  Transform data and preprocess  #########################################v
    XpredtestList = np.concatenate(XpredtestList, axis = 1)
    Ypredtest = C.dot(XpredtestList)
    
    ## Now we transpose things back to our normal Nsamxdim
    Matrix_tr = Matrix_tr.T
    Yhat = Yhat.T
    Ypred = Ypred.T
    Ypredtest = Ypredtest.T
    diff = diff.T
    
    sigmaXhatList = ul.fnp([ x[0,0] for x in SigmaXXhatList])
    sigmaXpredList= ul.fnp([ x[0,0] for x in SigmaXXpredList])
    sigmaXpredtestList = ul.fnp([ x[0,0] for x in SigmaXXpredtestList])
    
    sigmaXhatList = np.sqrt(sigmaXhatList )
    SigmaXpredList = np.sqrt(sigmaXpredList)
    
    # Now we plot
    RealPrice = Matrix_tr[:,[0]]
    EstimatedPrice = Yhat[:,[0]]
    PredictedPrice = Ypred[:,[0]]
    PredictedPriceTest = Ypredtest[:,[0]]
#    gl.plot([], diff, legend = ["Estate"])
    
    # Plot the training data and the estimated state
    gl.scatter(dates, RealPrice, legend = ["Real"])
    gl.plot_timeSeriesRange(dates, EstimatedPrice,sigmaXhatList, legend = ["Estate"], nf = 0)
    
    # Plot the one Step prediction
    dates_OneStepPred = ul.fnp(range(0, dates.size + 1))
    gl.plot_timeSeriesRange(dates_OneStepPred, PredictedPrice[:,:],sigmaXpredList[:,:], legend = ["Prediction"], nf = 0)
    
    ## Plot the future prediction
    dates_test = ul.fnp(range(dates.size, dates.size + Ntest))
    gl.plot_timeSeriesRange(dates_test, PredictedPriceTest[:-1,:],sigmaXpredtestList[:-1,:], legend = ["Prediction future"], nf = 0)

    # Using the state formulate, plot the prediction line from each state point.
    for n in range (0,Ns):
        start_point = XhatList[:,[n]]
        end_point = A.dot(start_point)
        
        gl.plot([dates_OneStepPred[n], dates_OneStepPred[n+1]],[start_point[0,0], end_point[0,0]],
                nf = 0, color = "k")
        
    ###########################################################################

# TODO: Obtener las mejores componented de una serie, hacemos PCA y detransformamos ? 
# Utiliar forma de relacionar las time series para crear muchos puntos que luego el GP 
# obtenga la senal inicial ?
# Usar la piFilter library y los processos gaussianos de sklearn
# TODO: Use this df["date"] = pd.to_datetime(df.index)
# Perform module for linear regression as in Time Series. 
# Estimation and Prediction of Variance and True Value.
# Use ARMA models. 
# Gaussian Process
# HP filtering 
# Normal Averages and BB.
# Kalman Filter
# Discrete frequency filters.
# TODO: Use the new means and Volatilies to construct new MACDs ?
# Train future systems with the smoothed versions ? 

# TODO: Event detection, like huge drops and so on, with the Drawdown ?