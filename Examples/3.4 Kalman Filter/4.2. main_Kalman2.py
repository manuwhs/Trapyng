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
import basicMathlib as bMA
import utilities_lib as ul
import KalmanFilter as KF
plt.close("all") # Close all previous Windows
######## SELECT DATASET, SYMBOLS AND PERIODS ########
dataSource =  "GCI"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)

#symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
#symbols = ["Amazon"]
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
    returns  = bMA.diff(timeSeries, cval = 0.000001)
    
    dates = timeData.get_dates()
    dates = ul.fnp(ul.datesToNumbers(dates))
    dates = ul.fnp(range(dates.size))
        
    ############################ Create the data #######################################
    Matrix_tr = np.concatenate([timeSeries,returns], axis = 1)  # Nsam * 4
    Ns,Nd = Matrix_tr.shape
    ########################### AR matrix ###############################
    # We assume as simple relationship
    # P_t = P_(t-1) + vP_(t-1)  -> The dinamics of the next price
    # vxt = vx_(t-1)            -> The dinamics of the next return
    A = np.array([[1,0.99],
                  [0,1]])
    ######################### C Matrix ###################################
    # Measurement matrix.
    # What transformation of the real state we perfrom.
    # In this case we just select both variables indpendently 
    C = np.eye(Nd)    # np.array([[1,1],[0,1]])
    
    ######################### B Matrix ###################################
    # In our case B*Ut is 0, we do not have acceleration modeled.
    B = None
    ################# Noise Paramters. Covariance Matrixes ###################  
    # Parameters
    varPrice = 5
    varDiffPrice = 2# Because it is the substraction of 2 random variables of variance 1
    varNoisePrice =  20 # Variance of the observation noise.
    varNoiseDiff = 10  # Variance of the observation noise.
    
    # Dynamic System error Covariance Matrix
    # Covariance noise of the System
    # The error that ocurrs in the system due to variables that we are not taking
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
    

    ######################################################################
    ############ USE THE KF !! #######
    ######################################################################
    Ntst = 10
    myKF = KF.KalmanFilter(A = A,B = B,C = C,
                           SystemCovNoise = SystemCovNoise, MeasurCovNoise = MeasurCovNoise)
    Yhat,SigmaXXhatList,Ypred, SigmaXXpredList = myKF.fit(dates, Matrix_tr)
    Ypredtest,SigmaXXpredtestList = myKF.predict(Ntst = Ntst)
    
    ######################################################################
    ## Extract the data to plot
    RealPrice = Matrix_tr[:,[0]]
    EstimatedPrice = Yhat[:,[0]]
    PredictedPrice = Ypred[:,[0]]
    
    sigmaXhatList = ul.fnp([ x[0,0] for x in SigmaXXhatList])
    sigmaXpredList= ul.fnp([ x[0,0] for x in SigmaXXpredList])
    sigmaXhatList = np.sqrt(sigmaXhatList )
    SigmaXpredList = np.sqrt(sigmaXpredList)
    
    sigmaYhatList = np.sqrt(sigmaXhatList**2 + MeasurCovNoise[0,0])
    
    PredictedPriceTest = Ypredtest[:,[0]]
    sigmaXpredtestList = ul.fnp([ x[0,0] for x in SigmaXXpredtestList])
    sigmaXpredtestList = np.sqrt(sigmaXpredtestList)

    # Plot the training data and the estimated state
    gl.scatter(dates, RealPrice, legend = ["Real"])
    gl.plot_timeSeriesRange(dates, EstimatedPrice,sigmaXhatList, legend = ["Estate"], nf = 0)
    gl.plot_timeSeriesRange(dates, EstimatedPrice,sigmaYhatList, legend = ["Estate"], nf = 0)
    
    # Plot the one Step prediction
    dates_OneStepPred = ul.fnp(range(0, dates.size + 1))
    gl.plot_timeSeriesRange(dates_OneStepPred, PredictedPrice[:,:],SigmaXpredList[:,:], legend = ["Prediction"], nf = 0)
#    
    ## Plot the future prediction
    dates_test = ul.fnp(range(dates.size, dates.size + Ntst +1))
    gl.plot_timeSeriesRange(dates_test, PredictedPriceTest[:,:],sigmaXpredtestList[:,:], legend = ["Prediction future"], nf = 0)
    
    ######################################################################
    ## Optimization !!! 

    xopt = myKF.optimize_parameters(100,100, 100)
    
    SystemCovNoise, MeasurCovNoise = myKF.build_CovMatrix(Matrix_tr, xopt[0], xopt[1],xopt[2])
    
   ######################################################################
    ############ USE THE KF !! #######
    ######################################################################
    Ntst = 10
    myKF = KF.KalmanFilter(A = A,B = B,C = C,
                           SystemCovNoise = SystemCovNoise, MeasurCovNoise = MeasurCovNoise)
    Yhat,SigmaXXhatList,Ypred, SigmaXXpredList = myKF.fit(dates, Matrix_tr)
    Ypredtest,SigmaXXpredtestList = myKF.predict(Ntst = Ntst)
    
    ######################################################################
    ## Extract the data to plot
    RealPrice = Matrix_tr[:,[0]]
    EstimatedPrice = Yhat[:,[0]]
    PredictedPrice = Ypred[:,[0]]
    
    sigmaXhatList = ul.fnp([ x[0,0] for x in SigmaXXhatList])
    sigmaXpredList= ul.fnp([ x[0,0] for x in SigmaXXpredList])
    sigmaXhatList = np.sqrt(sigmaXhatList )
    SigmaXpredList = np.sqrt(sigmaXpredList)
    
    sigmaYhatList = np.sqrt(sigmaXhatList**2 + MeasurCovNoise[0,0])
    
    PredictedPriceTest = Ypredtest[:,[0]]
    sigmaXpredtestList = ul.fnp([ x[0,0] for x in SigmaXXpredtestList])
    sigmaXpredtestList = np.sqrt(sigmaXpredtestList)

    # Plot the training data and the estimated state
    gl.scatter(dates, RealPrice, legend = ["Real"])
    gl.plot_timeSeriesRange(dates, EstimatedPrice,sigmaXhatList, legend = ["Estate"], nf = 0)
    gl.plot_timeSeriesRange(dates, EstimatedPrice,sigmaYhatList, legend = ["Estate"], nf = 0)
    
    # Plot the one Step prediction
    dates_OneStepPred = ul.fnp(range(0, dates.size + 1))
    gl.plot_timeSeriesRange(dates_OneStepPred, PredictedPrice[:,:],SigmaXpredList[:,:], legend = ["Prediction"], nf = 0)
#    
    ## Plot the future prediction
    dates_test = ul.fnp(range(dates.size, dates.size + Ntst +1))
    gl.plot_timeSeriesRange(dates_test, PredictedPriceTest[:,:],sigmaXpredtestList[:,:], legend = ["Prediction future"], nf = 0)
    
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


# TODO: Ideas:
# - En intradia o tal solo ver maximos y minimos
# - Ver anomalias de laterales respecto a volumen, si de repente los extremos tienen anomalias de volumen, entonces dale.
# - Sistema de cumulacion de seguridad, es decir, sistemas que predigan que algo raro va a pasar,
# - como mucha sobrecompra, convergencia de dinero, anomalias, y que avise ante inminente cambio de precio.