""" USE OF WIDGETS """ 
# Change main directory to the main folder and import folders
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

pandas_lib1 = 1
if (pandas_lib1 == 1):

###########################################################################
# Oscillators 

    STO = timeData.STO(n = 14)
    RSI = timeData.RSI(n = 14)
    ADX = timeData.ADX()
    ACCDIST = timeData.ACCDIST()
    
    gl.set_subplots(3,1)
    gl.plot(dates, price , nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price"])
            
    gl.plot(dates, STO, nf = 1, na = 0,
            legend = ["STO"])
    gl.plot(dates,  RSI , nf = 0, na = 0,
            legend = [ "RSI"])
    gl.plot(dates, ADX , nf = 0, na = 1,
            legend = ["ADX"])
            
    gl.plot(dates, ACCDIST , nf = 1, na = 0,
            legend = ["ACCDIST"])
#    ws = 40
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])


trading_station = 0
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


widgets_flag = 1
if (widgets_flag == 1):
    timeSeries = timeData.get_timeSeries(["High"]);
    dates = timeData.dates
    dates = convert_dates_str(dates)
#    dates = range(len(dates))
#    gl.plot_wid(timeData.get_dates(), timeSeries, scrolling = 200)

    ws = 50
    gl.step(dates,timeSeries, 
            legend =  ["price"], 
            ws = ws)
    
    timeSeries = timeData.get_timeSeries(["Low"]);
    gl.step(dates,timeSeries, 
            legend =  ["price"], 
            ws = ws,
            color = "red",
            nf = 0)
            
            
    timeSeries = timeData.get_timeSeries(["Volume"]);
#    gl.step(timeData.dates,timeSeries, 
#            legend =  ["price"], 
#            ws = ws,
#            color = "red",
#            nf = 0, na = 1, fill = 1)
            
    gl.bar(dates,timeSeries, 
            legend =  ["price"], 
            ws = ws,
            color = "red",
            nf = 0, na = 1)
            
    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
    gl.add_hidebox()
    
    list_values = []
    gl.add_selector(list_values)
    gl.add_onKeyPress()
#    type(timeData.TD.index)
#    cad = pd.DataFrame(index = time_index)
#    caca = pd.to_datetime(time_index.T.tolist())
#    print caca
#    


#    ld = CGraph()
#    ld = copy.copy(gl)
#
#    ws = 50
#    ld.plot(timeData.dates,timeSeries, 
#            legend =  ["price"], 
#            ws = ws)
#    
#    timeSeries = timeData.get_timeSeries(["Close"]);
#    ld.plot(timeData.dates,timeSeries, 
#            legend =  ["price"], 
#            ws = ws,
#            color = "red",
#            nf = 0)
#    
#    ld.bar(timeData.dates,timeSeries, 
#            legend =  ["price"], 
#            ws = ws,
#            color = "red",
#            nf = 0, na = 1)
#            
#    ld.add_slider(args = {"wsize":ws}, plots_affected = [])
#    ld.add_hidebox()
#    
    
#    
#    