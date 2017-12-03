import os
os.chdir("../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
# Data Structures Data
import CPortfolio as CPfl
import CSymbol as CSy
import CFilter as CFi
# Own graphical library
from graph_lib import gl 
# Import functions independent of DataStructure
import utilities_lib as ul
import pandas as pd
import basicMathlib as bMA
import indicators_lib as indl

plt.close("all")
######## SELECT DATASET, SYMBOLS AND PERIODS ########
source = "GCI" # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = source)
################## Date info ###################
sdate_str = "01-8-2014"
edate_str = "21-12-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
#### Load the info about the available symbols #####
load_info = 0
if (load_info == 1):
    Symbol_info = CSy.load_symbols_info(info_folder)
    Symbol_names = Symbol_info["Symbol_name"].tolist()
    Nsym = len(Symbol_names)
####################################################
####### SELECT SYMBOLS AND PERIODS #################
symbols = ["Alcoa_Inc", "Apple_Comp.","Ford_Motor", "Amazon"]
periods = [1440]
period = periods[0]
####### LOAD SYMBOLS AND SET    ###################
Cartera = CPfl.Portfolio(symbols, periods)   # Set the symbols and periods to load
Cartera.load_symbols_csv(storage_folder)    # Load the symbols and periods
## SET THINGS FOR ALL OF THEM
Cartera.set_interval(sdate,edate)
Cartera.set_seriesNames(["Close"])
######################## FILLING DATA  ##########################
print "Filling Data"
Cartera.fill_data()
print "Data Filled"
# TODO: Define better the get_timeSeries and dates for Cartera
prices = Cartera.get_timeSeries(symbol_names = [symbols[1]])
prices = Cartera.get_timeSeries()
dates = Cartera.get_dates(period, Cartera.symbol_names[0])

########################  OBTAIN BASIC FEATURES ##########################
## In this trials, the output will be the return lagged a given number of 
## time-instances.
# We get now the moving averages
indx = 1   # The Symbol we are gonna obtain the parameters from
lag = 10   # Lag of the return

## Price sequence that we want to estimate
price = prices[:,[indx]]
Y_data = bMA.diff(price, lag = lag, cval = np.NaN)
Y_data = bMA.shift(Y_data, lag = -lag, cval = np.NaN)
return_Ydata = bMA.get_return(price, lag = lag)
Y_data = return_Ydata
Y_data = bMA.shift(Y_data, lag = -lag, cval = np.NaN)
#Y_data = np.sign(Y_data)
#Y_data = bMA.shift(prices[:,indx], lag = 5, cval = np.NaN)

## MACD Parameters !!
n_fast = 12; n_slow = 26; n_smooth = 9
EMAfast = Cartera.EMA(n = n_fast)
EMAslow = Cartera.EMA(n = n_slow)
MACD = EMAfast - EMAslow
MACD = pd.ewma(MACD, span = n_smooth, min_periods = n_smooth - 1)
## Smoothin of velocity, we want to enter when it is falling
MACD_vel = bMA.diff(MACD,n = 1)
nsmooth_vel = 4
MACD_vel = indl.get_SMA(MACD_vel, L = nsmooth_vel)
## RSI Parameter !
RSI = Cartera.RSI(n = 14)
RSI_vel = bMA.diff(RSI,n = 1)
#RSI_vel = indl.get_SMA(RSI_vel, L = nsmooth_vel)
## ATR Parameter !
ATR = Cartera.ATR(n = 14)
ATR_vel = bMA.diff(ATR,n = 1)
RSI_vel = indl.get_SMA(ATR_vel, L = nsmooth_vel)

################# PREPARE THE DATA MATRIX FORM ########################
X_data = np.concatenate((MACD[:,[indx]],MACD_vel[:,[indx]]), axis = 1)
X_data = np.concatenate((X_data,RSI[:,[indx]], ATR[:,[indx]]), axis = 1)
X_data = np.concatenate((X_data,RSI_vel[:,[indx]], ATR_vel[:,[indx]]), axis = 1)

################# PREPARE THE PANDAS FORM ########################
# Some algorihtms expect the data in this form
data = pd.DataFrame(
{'Date': dates[0], 'MACD': MACD[:,indx], 'MACD_vel':  MACD_vel[:,indx], 'ATR' : ATR[:,indx], 
'RSI': RSI[:,indx], 'RSI_vel': RSI_vel[:,indx], 'ATR_vel': ATR_vel[:,indx], 
'Price':price[:,0], 'Y':  Y_data[:,0]})
data.set_index (["Date"], inplace = True)

################# SAVE THE DATA TO DISK ########################
folder_dataFeatures = "./data/"
data.to_csv(folder_dataFeatures+ "dataPandas.csv")
np.savetxt(folder_dataFeatures + "Xdata.csv", X_data, delimiter=",")
np.savetxt(folder_dataFeatures + "price.csv", price, delimiter=",")
np.savetxt(folder_dataFeatures + "dates.csv", dates[0], delimiter=",")
## TODO: Put somewhere else
####### RECONSTRUCTION PROOF OF THE DATA ############## 
reconstruct_data = 0
if reconstruct_data:
    lag_ret = 20
    return_Ydata = bMA.get_return(price, lag = lag_ret)
    reconstruct_Ydata = bMA.reconstruc_return(price, return_Ydata, lag = lag_ret)
    
    gl.plot([],price, legend= ["price"])
    gl.plot([],reconstruct_Ydata, nf = 0, legend= ["reconstruction"])
    gl.plot([],return_Ydata, nf = 0, na = 1, legend= ["return"])

### Filter the data according to some parameter
filter_data = 1
if (filter_data):
    myFilter = CFi.CFilter()
    mascara = myFilter.get_ThresholdMask(abs(X_data[:,1]), ths = [0.2], reverse = True)
    gl.plot([], abs(X_data[:,1]))
    gl.plot([], np.ones(X_data[:,1].shape) * 0.1, nf = 0)
    gl.plot([], np.ones(X_data[:,1].shape) * 0.3, nf = 0)
    gl.plot([], mascara, nf = 0, na = 1, fill = 1, alpha = 0.3)

#    ## Redo another selection
#    mask_sel = np.argwhere(abs(X_data[:,1]) < 0.1)[:,0]
#    #mask_sel_neg = np.argwhere(X_data[:,0] > -4)[:,0]

    X_data = myFilter.apply_Mask(X_data, mascara)
    Y_data = myFilter.apply_Mask(Y_data, mascara)

################################################################
###################  PLOTTING THE DATA #########################
################################################################
## TODO: idea, a high variance, indicates consinuity in the parameters
## Plotting plotting
plotting_some = 1
if (plotting_some == 1):
    Ndiv = 6; HPV = 2
    ### PLOT THE ORIGINAL PRICE AND MOVING AVERAGES
    timeData = Cartera.get_timeDataObj(period = -1, symbol_names = [symbols[indx]])[0]
    
    gl.init_figure()
    ##########################  AX0 ########################
    ### Plot the initial Price Volume
    gl.subplot2grid((Ndiv,4), (0,0), rowspan=HPV, colspan=4)
#    gl.plot_indicator(timeData, Ndiv = Ndiv, HPV = HPV)
    gl.plot(dates, prices[:,indx], legend = ["Price"], nf = 0,
            labels = ["Xing Average Strategy","Price","Time"])
    
    Volume = timeData.get_timeSeries(seriesNames = ["Volume"])
    gl.plot(dates, Volume, nf = 0, na = 1, lw = 0, alpha = 0)
    gl.fill_between(dates, Volume)
    
    axes = gl.get_axes()
    axP = axes[0]  # Price axes
    axV = axes[1]  # Volumne exes
    
    axV.set_ylim(0,3 * max(Volume))
    gl.plot(dates, EMAfast[:,indx],ax = axP, legend = ["EMA = %i" %(n_fast)], nf = 0)
    gl.plot(dates, EMAslow[:,indx],ax = axP, legend = ["EMA = %i" %(n_slow)], nf = 0)
    
    ##########################  AX1 ########################
    pos = 0
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    gl.plot(dates, Y_data, nf = 0, legend = ["Y data filtered"])
    
   ##########################  AX2 ########################
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    
    gl.plot(dates, MACD[:,indx], nf = 0, legend = ["MACD"])
    gl.plot(dates, MACD_vel[:,indx], nf = 0, na = 1, legend = ["MACD_vel"])
    
    ##########################  AX3 ########################
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    gl.plot(dates, RSI[:,indx], nf = 0, legend = [ "RSI"])
    gl.plot(dates, RSI_vel[:,indx], nf = 0, na = 1, legend = [ "RSI_vel"])
    ##########################  AX4 ########################
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)

#    BB = timeData.BBANDS(seriesNames = ["Close"], n = 14)
#    gl.plot(dates,  BB[:,0] - BB[:,1] , nf = 0, na = 0, legend = [ "BB"])
    gl.plot(dates, ATR[:,indx], nf = 0,legend = ["ATR"])
    gl.plot(dates, ATR_vel[:,indx], nf = 0, na = 1,legend = ["ATR_vel"])

    ## Format the axis
    all_axes = gl.get_axes()
    for i in range(len(all_axes)-2):
        ax = all_axes[i]
        plt.setp(ax.get_xticklabels(), visible=False)
    
    gl.format_axis2(all_axes[-2], Nx = 20, Ny = 5, fontsize = -1, rotation = 45)
    plt.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)
    
    
plot_others = 0
if(plot_others):
    Xdata = np.array([MACD[:,indx], RSI[:,indx], MACD_vel[:,indx]]).T
    ## Graoca bonita !! TODO
    print ul.fnp(np.sqrt(np.sum(Xdata*Xdata, axis = 1))).shape
    Xdata = Xdata / ul.fnp(np.sqrt(np.sum(Xdata*Xdata, axis = 1)))
    gl.scatter_3D(Xdata[:,0],Xdata[:,1],Xdata[:,2])
    
    gl.scatter_3D(-Xdata[:,0],-Xdata[:,1],-Xdata[:,2], nf = 0)
    gl.scatter_3D(-Xdata[:,0],Xdata[:,1],-Xdata[:,2], nf = 0)
    gl.scatter_3D(Xdata[:,0],-Xdata[:,1],Xdata[:,2], nf = 0)
    
    # TODO: watch time evoultion also of 2 variables with time