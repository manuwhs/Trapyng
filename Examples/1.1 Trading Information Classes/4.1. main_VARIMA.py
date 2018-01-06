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


VARMA_f = 0
advanced_smoothing_f = 0
KM = 0
Gaussin_Process = 1

############## Classic TimeSeries Analysis ##################
if (VARMA_f == 1):
    timeSeries = timeData.get_timeSeries(["Average"]);
    returns = timeData.get_timeSeriesReturn()
    grtsa.plot_acf_pacf(returns[:,0])
    grtsa.plot_decomposition(timeSeries[:,0].T)
    
############## TIME SERIES INDICATORS #####################################
if (advanced_smoothing_f == 1):
    price = timeData.get_timeSeries(["Average"]);
    casd = ul.get_Elliot_Trends(price,10);
    timeData.plot_timeSeries()
    flag_p = 0
    for trend in casd:
        gl.plot(timeData.dates[trend], price[trend], lw = 5, nf = flag_p)
        flag_p = 0


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