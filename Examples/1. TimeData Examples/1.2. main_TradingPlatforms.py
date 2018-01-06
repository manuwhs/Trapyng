""" MORE PROFESSIONAL PLOTTINGS OF THE TRADING INDICATORS"""
# Change main directory to the main folder and import folders


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

trading_indicators_complex = 1
if (trading_indicators_complex == 1):
    ## Example on how to construct complex graphs showing better graphs
    dates = timeData.get_dates()
    Ndiv = 6; HPV = 2
    
    ### Plot the initial Price Volume
    gl.plot_indicator(timeData, Ndiv = Ndiv, HPV = HPV)
    axes = gl.get_axes()
    axP = axes[0]  # Price axes
    axV = axes[1]  # Volumne exes
    
    # Plot something on the price
    priceA = timeData.get_timeSeries(["Average"]);
    MA1 = 16; MA2 = 9
    Av1 = timeData.SMA( n = MA1)
    Av2 = timeData.SMA(n = MA2)
    
    gl.plot(dates,Av1, nf = 0, ax = axP,
            color = '#e1edf9',legend=[str(MA1)+' SMA'], lw = 1.5)
    gl.plot(dates,Av2, nf = 0, ax = axP,
            color = '#4ee6fd',legend=[str(MA2)+' SMA'], lw = 1.5)
    gl.plot(dates, priceA, ax = axP,nf = 0, na = 0,
            legend = ["Average price"])
    # Formating the legend
    maLeg = axP.legend(loc=9, ncol=2, prop={'size':7},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = axP.get_legend().get_texts()
    pylab.setp(textEd[0:5], color = 'w')
    
    ## Plot other indicators
    pos = 0
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP, axisbg='#07000d')
    gl.plotMACD(timeData, ax = None)
    
    ## Plot other indicators
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    
    MOM = timeData.MOM(n = 1)
    ROC = timeData.ROC(n = 2)
    gl.plot(dates, MOM , nf = 0, na = 0, legend = ["Momentum"])
    gl.plot(dates, ROC , nf = 0, na = 1, legend = ["ROC"])
            
    ## Plot more indicators
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)

    STO = timeData.STO(n = 14)
    RSI = timeData.RSI(n = 14)
    ADX = timeData.ADX()
    
    gl.plot(dates, STO, nf = 0, na = 0, legend = ["STO"])
    gl.plot(dates,  RSI , nf = 0, na = 0, legend = [ "RSI"])
    gl.plot(dates, ADX , nf = 0, na = 1,legend = ["ADX"])
    
    ## Plot more indicators
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)

    BB = timeData.BBANDS(seriesNames = ["Close"], n = 10)
    ATR = timeData.ATR(n = 20)
    
    gl.plot(dates,  BB[:,0] - BB[:,1] , nf = 0, na = 0, legend = [ "BB"])
    gl.plot(dates, ATR , nf = 0, na = 1,legend = ["ATR"])
    
    ## Format the axis
    all_axes = gl.get_axes()
    for i in range(len(all_axes)-2):
        ax = all_axes[i]
        plt.setp(ax.get_xticklabels(), visible=False)
    
    gl.format_axis2(all_axes[-2], Nx = 20, Ny = 5, fontsize = -1, rotation = 45)
    plt.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)


trading_station = 1
if (trading_station == 1):
    gl.init_figure()
    gl.tradingPlatform(timeData,volumeFactor = 1)

    # Define the axes as we want or...
#    marginL = 0.05
#    marginB = 0.05
#    trading_pos = [0.05,0.05,0.9,0.35]
#    low_ax_pos2 = [0.05,0.05,0.9,0.3501]    
#    
#    gl.init_figure()
##    gl.create_axes(position = [0.05,0.05, 0.8, 0.8])
#    ax1 = gl.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)
#    gl.tradingPV(timeData)
#    
#    ax2 = gl.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
#    RSI = timeData.RSI(n = 14)
#    gl.tradingOcillator(timeData, RSI)
#    
#    ax3 = gl.subplot2grid((5,4), (4,0), rowspan=1, colspan=4)
#    STO = timeData.STO(n = 14)
#    gl.tradingOcillator(timeData, STO)
#    
#    ws = 100  
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
#    gl.add_hidebox()
    
    plt.subplots_adjust(left=.09, bottom=.14, right=.90, top=.95, wspace=.0, hspace=0)

#############################################################
############## Classic TimeSeries Analysis #################################
#############################################################
tsa_f = 0
if (tsa_f == 1):
    timeSeries = timeData.get_timeSeries(["Average"]);
    returns = timeData.get_timeSeriesReturn()
    grtsa.plot_acf_pacf(returns[:,0])
    grtsa.plot_decomposition(timeSeries[:,0].T)
