""" EXPERIMENTAL INDICATORS """
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
sdate_str = "15-01-2016"
edate_str = "2-6-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Tell which company and which period we want
timeData = CTD.CTimeData(symbols[0],periods[0])
timeData.set_csv(storage_folder)  # Load the data into the model
timeData.set_interval(sdate,edate) # Set the interval period to be analysed

###########################################################################
############## Pandas indicator Library ############################################

BB_f = 1
AHLR_f = 1
ATR_f = 1
chaikin_f = 1

folder_images = "../pics/Trapying/Volatility/"
if (BB_f):

    dataHLOC = timeData.get_timeSeries(["High","Low","Open","Close"])
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()

   # Bollinger Bands and ATR
    nBB = 10; nBB1 = 20; nBB2 = 10
    
    SMA1 = timeData.SMA(n = nBB1)
    SMA2 = timeData.SMA(n = nBB2)
    STD1 = timeData.STD(n = nBB1)
    STD2 = timeData.STD(n = nBB2)
    
    color1 = "k"
    colorBB1 = "cobalt blue"
    colorBB2 = "irish green"
    ## Plotting t the BB !!
    ax1 = gl.subplot2grid((5,1), (0,0), rowspan=4, colspan=1)
#    gl.plot(dates, price, ax = ax1,  AxesStyle = "Normal - No xaxis",
#            labels = ["Volatility indicator BB", "", "Price"],
#            legend = ["Price"], color = "dark navy blue")
    title = "Bollinger Bands. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
    gl.barchart(dates, dataHLOC, ax = ax1, color = color1,
            labels = [title, "", r"Price ($\$$)"],
            legend = ["HLOC Price"])
            
    gl.plot(dates, SMA1, color = colorBB1,
            legend = ["SMA(%i)"%nBB1])
    gl.plot(dates, SMA2 , color = colorBB2,AxesStyle = "Normal - No xaxis",
            legend = [ "SMA(%i)"%nBB2])

    ## 
    ax2 = gl.subplot2grid((5,1), (4,0), rowspan=1, colspan=1, sharex = ax1)
    gl.plot(dates, STD1 ,ax = ax2, color =  colorBB1,
            labels = ["","","STD"],  
            legend = ["STD(%i)"%nBB1], fill = 1, alpha = 0.3)
            
    gl.plot(dates, STD2, ax = ax2,
             color = colorBB2,  AxesStyle = "Normal - Ny:5",
             legend = ["STD(%i)"%nBB2], fill = 1, alpha = 0.5
             )
#            
#    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)
#
#    gl.savefig(folder_images +'VolatilitySTD.png', 
#               dpi = 100, sizeInches = [2*8, 2*3])
#               
##################### ATR ######################################    
if (AHLR_f):
    dataHLOC = timeData.get_timeSeries(["High","Low","Open","Close"])
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()

    nAHLR1 = 14; nAHLR2 = 20
    AHLR1 = timeData.AHLR(n = nAHLR1)
    AHLR2 = timeData.AHLR(n = nAHLR2)
    SMA1_AHLR = timeData.SMA(n = nAHLR1)
    SMA2_AHLR = timeData.SMA(n = nAHLR2)
    
    
    ## Plotting just the ATR !!
    color1 = "k"
    colorBB1 = "cobalt blue"
    colorBB2 = "irish green"
    gl.init_figure()
    ax1 = gl.subplot2grid((5,1), (0,0), rowspan=4, colspan=1)
#    gl.plot(dates, price,  nf = 1,
#            labels = ["Volatility indicator ATR", "", "Price"],
#            legend = ["Price"])
    title = "ATR. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
    gl.barchart(dates, dataHLOC, ax = ax1, color = color1,
            labels = [title, "", r"Price ($\$$)"],
            legend = ["HLOC Price"])
    gl.plot(dates, SMA1, color = colorBB1,
            legend = ["SMA(%i)"%nAHLR1])
    gl.plot(dates, SMA2 , color = colorBB2,AxesStyle = "Normal - No xaxis",
            legend = [ "SMA(%i)"%nAHLR2])
    ## 2
    ax2 = gl.subplot2grid((5,1), (4,0), rowspan=1, colspan=1, sharex = ax1)
    gl.plot(dates, AHLR1 ,ax = ax2, color = colorBB1,
            labels = ["","","AHLR"],
            legend = ["AHLR(%i)"%nAHLR1], fill = 1, alpha = 0.3)
    gl.plot(dates, AHLR2, ax= ax2,
            color = colorBB2, AxesStyle = "Normal - Ny:5",
            legend = ["AHLR(%i)"%nAHLR2], fill = 1, alpha = 0.5)
            
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

    gl.savefig(folder_images +'VolatilityAHLR.png', 
               dpi = 100, sizeInches = [2*8, 2*4])
               
##################### ATR ######################################    
if (ATR_f):
    dataHLOC = timeData.get_timeSeries(["High","Low","Open","Close"])
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()
    df = timeData.get_TD()

    nAHLR1 = 14; nAHLR2 = 20
    AHLR1 = timeData.AHLR(n = nAHLR1)
    AHLR2 = timeData.AHLR(n = nAHLR2)
    SMA1_AHLR = timeData.SMA(n = nAHLR1)
    SMA2_AHLR = timeData.SMA(n = nAHLR2)
    
    nATR1 = 14; nATR2 = 20
    ATR1 = timeData.ATR(n = nATR1)
    ATR2 = timeData.ATR(n = nATR2)
    SMA1_ATR = timeData.SMA(n = nATR1)
    SMA2_ATR = timeData.SMA(n = nATR2)
    
    ## Plotting just the ATR !!
    color1 = "k"
    colorBB1 = "cobalt blue"
    colorBB2 = "irish green"
    gl.init_figure()
    ax1 = gl.subplot2grid((5,1), (0,0), rowspan=3, colspan=1)
#    gl.plot(dates, price,  nf = 1,
#            labels = ["Volatility indicator ATR", "", "Price"],
#            legend = ["Price"])
    title = "ATR. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
    gl.barchart(dates, dataHLOC, ax = ax1, color = color1,
            labels = [title, "", r"Price ($\$$)"],
            legend = ["HLOC Price"])
    gl.plot(dates, SMA1, color = colorBB1,
            legend = ["SMA(%i)"%nATR1])
    gl.plot(dates, SMA2 , color = colorBB2,AxesStyle = "Normal - No xaxis",
            legend = [ "SMA(%i)"%nATR2])

    ## 3
    ax3 = gl.subplot2grid((5,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    gl.plot(dates, ATR1 ,ax = ax3, color = colorBB1,
            labels = ["","","ATR"],
            legend = ["ATR(%i)"%nATR1], fill = 1, alpha = 0.3)
    gl.plot(dates, ATR2, ax= ax3,
             color = colorBB2, AxesStyle = "Normal - Ny:5 - No xaxis",
            legend = ["ATR(%i)"%nATR2], fill = 1, alpha = 0.5)
            
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

    gl.savefig(folder_images +'VolatilityATR.png', 
               dpi = 100, sizeInches = [2*8, 2*5])

    ## 2
    ax2 = gl.subplot2grid((5,1), (4,0), rowspan=1, colspan=1, sharex = ax1)
    gl.plot(dates, ATR1 - AHLR1 ,ax = ax2, color = colorBB1,
            labels = ["","","ATR - AHLR"],
            legend = ["diff(%i)"%nAHLR1], fill = 1, alpha = 0.3)
    gl.plot(dates, ATR2 - AHLR2, ax= ax2,
             color = colorBB2, AxesStyle = "Normal - Ny:5",
            legend = ["diff(%i)"%nAHLR2], fill = 1, alpha = 0.5)
            
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

    gl.savefig(folder_images +'VolatilityATR.png', 
               dpi = 100, sizeInches = [2*8, 2*2])
                 
##################### Chaikin ######################################
 
if (chaikin_f):
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()
    
    nCha1 = 14
    nCha2 = 20
    EMA_Range1, Cha1 = timeData.Chaikin_vol(n = nCha1)
    EMA_Range2,Cha2 = timeData.Chaikin_vol(n = nCha2)
    SMA1_ATR = timeData.EMA(n = nCha1)
    SMA2_ATR = timeData.EMA(n = nCha2)
    
    
    ## Plotting just the ATR !!
    # TODO: Divide 3-1
    gl.set_subplots(3,1)
    gl.plot(dates, price,  nf = 1,
            labels = ["Volatility indicator Chainkin", "", "Price"],
            legend = ["Price"])
            
    gl.plot(dates, SMA1_ATR , nf = 0, color = "k",
            legend = ["Chainkin(%i)"%nATR1])
    gl.plot_filled(dates, np.concatenate([SMA1_ATR + 2*EMA_Range1,SMA1_ATR - 2*EMA_Range1],axis = 1), 
                   alpha = 0.5, nf = 0, color = "k")

    gl.plot(dates, SMA2_ATR , nf = 0, color = "y",
            legend = [ "Chainkin(%i)"%nATR2])
            
    gl.plot_filled(dates, np.concatenate([SMA2_ATR + 2*EMA_Range2,SMA2_ATR - 2*EMA_Range2],axis = 1), 
                   alpha = 0.5, nf = 0, color = "y")
    
    
    gl.plot(dates, EMA_Range1 , nf = 1, na = 0, color = "k",
            labels = ["","","Volatility"],
            legend = ["EMA_HL(%i)"%nCha1], fill = 1, alpha = 0.5)
    gl.plot(dates, EMA_Range2 , nf = 0, na = 0, color = "y",
            legend = ["EMA_HL(%i)"%nCha2], fill = 1, alpha = 0.5)
    gl.plot(dates, np.zeros((dates.size,1)) , nf = 0, na = 0)
             
    ## 2
    gl.plot(dates, Cha1 , nf = 1, na = 0, color = "k",
            labels = ["","","Change in Volatility"],
            legend = ["Chaikin(%i)"%nCha1], fill = 1, alpha = 0.5)
    gl.plot(dates, Cha2 , nf = 0, na = 0,color = "y",
            legend = ["Chaikin(%i)"%nCha2], fill = 1, alpha = 0.5)
            
    # The nect plot is just so that the vision starts in the first date
    gl.plot(dates, np.zeros((dates.size,1)) , nf = 0, na = 0)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

    gl.savefig(folder_images +'VolatilityChaikin.png', 
               dpi = 100, sizeInches = [2*8, 2*3])
    
#    gl.set_subplots(3,1)
#    gl.plot(dates, [price, BB[:,0],BB[:,1]] , nf = 1,
#            labels = ["Volatility indicators ATR and BB", "", "Price"],
#            legend = ["Price", "BB(%i)"%nBB])
#            
#    gl.fill_between(x = dates, y1 = BB[:,0], y2 = BB[:,1], alpha = 0.5)
#
#    gl.plot(dates, [ BBE[:,0],BBE[:,1]] , nf = 0,
#            legend = ["BBE(%i)"%nBB])
#            
#    gl.fill_between(x = dates, y1 = BBE[:,0], y2 = BBE[:,1], alpha = 0.5)
#
#    ## 2
#    gl.plot(dates, price , nf = 1,
#            labels = ["","","Price"],
#            legend = ["Price"])
#            
#    gl.fill_between(dates, price - ATR, price + ATR, nf = 0, na = 0,
#            legend = ["ATR"], fill = 0)
#    ## 3
#    gl.plot(dates, ATR , nf = 1, na = 0,
#            labels = ["Averages","Time","Value"],
#            legend = ["ATR"], fill = 1, alpha = 0.5)
#    gl.plot(dates, BB[:,0] - BB[:,1] , nf = 0, na = 1,
#            labels = ["Averages","Time","Value"],
#            legend = ["BB"], fill = 1, alpha = 0.5)
#    gl.plot(dates, BBE[:,0] - BBE[:,1] , nf = 0, na = 1,
#            labels = ["Averages","Time","Value"],
#            legend = ["BB"], fill = 1, alpha = 0.5)
#            
#    gl.plot(dates, np.zeros((dates.size,1)) , nf = 0, na = 0,
#            labels = ["Averages","Time","Value"],
#            legend = ["BB"], fill = 1, alpha = 0.5)
#    
#    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)
#
#    gl.savefig(folder_images +'Volatility.png', 
#               dpi = 100, sizeInches = [2*8, 2*6])