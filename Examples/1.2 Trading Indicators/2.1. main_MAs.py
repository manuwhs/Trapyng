""" 
In this file we can plot different Moving Averages

"""
# Change main directory to the main folder and import folders
import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt

# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Import functions independent of DataStructure
import utilities_MQL5 as ul5
import basicMathlib as bMA
import indicators_lib as indl

plt.close("all") # Close all previous Windows

######## SELECT DATASET, SYMBOLS AND PERIODS ########
dataSource =  "MQL5"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, updates_folder] = ul5.get_foldersData(source = dataSource)

symbols = ["EURUSD"]
periods = [1440]
######## SELECT DATE LIMITS ###########
sdate_str = "01-02-2018"; edate_str = "2-2-2019"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Tell which company and which period we want
timeData = CTD.CTimeData(symbols[0],periods[0])
timeData.set_csv(storage_folder)  # Load the data into the model
timeData.set_interval(sdate,edate) # Set the interval period to be analysed

###########################################################################
############## Pandas indicator Library ############################################
###########################################################################
folder_images = "../pics/Trapying/MA/"
# Using the library of function built in using the dataFrames in pandas
comparing_SEW_MAs = 1
comparing_lags = 1
viewing_SEW_windows = 1
MAMAs = 1
HullsMA = 1

if (comparing_SEW_MAs):
    # Some basic indicators.
    price = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()
    
    # For comparing SMA, EMA, WMA
    nMA1 = 30
    # For comparing SMA, EMA, WMA
    SMA1 = timeData.SMA(seriesNames = ["Close"], n = nMA1)
    EMA1 = timeData.EMA(seriesNames = ["Close"], n = nMA1)
    MWA1 = timeData.get_WMA(nMA1)
    # Plotting the 3 of them at the same time.
    title = "Comparing MAs. " + str(symbols[0]) + "(" + ul5.period_dic[timeData.period]+ ")" 
    gl.plot(dates, [price, SMA1, MWA1, EMA1], nf = 1,
            labels = [title,"",r"Price ($\$$)"],
            legend = [r"$P_{CLOSE}$", "SMA(%i)"%nMA1, "WMA(%i)"%nMA1, "EMA(%i)"%nMA1],
            AxesStyle = "Normal")
#        ls = "-", marker = ["*",5,None], fill= 1,AxesStyle = "Normal", alpha = 0.3)
    
    # TODO: Why is this one not shown ?
    gl.savefig(folder_images +'comparingMAs.png', 
               dpi = 100, sizeInches = [10, 6])

if (comparing_lags):
    # Some basic indicators.
    price = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()
    
    nSMAs = [7,20,50]
    nEMAs = [7,20,50]
    # For lag and noise
    SMAs = []
    for nMA_i in nSMAs:
        SMAs.append(timeData.SMA(seriesNames = ["Close"], n = nMA_i))
    EMAs = []
    for nMA_i in nEMAs:
        EMAs.append(timeData.EMA(seriesNames = ["Close"], n = nMA_i))
    ############## PLOTTING ################
    gl.set_subplots(2,1)
    # Axes with the SMAs
    legend = ["Price"]
    legend.extend([ "SMA(" +str(x) +")" for x in nSMAs])
    SMAs.insert(0, price)
    
    title = "Influence of L in the lag. " + str(symbols[0]) + "(" + ul5.period_dic[timeData.period]+ ")"
    ax1 = gl.plot(dates,  SMAs , nf = 1,
            labels = [title,"",r"Price ($\$$)"],
            legend = legend,  AxesStyle = "Normal - No xaxis")
    # Axes with the EMAs
    legend = ["Price"]
    EMAs.insert(0, price)
    legend.extend([ "EMA(" +str(x) +")"for x in nEMAs])
    gl.plot(dates, EMAs, nf = 1, sharex= ax1,
            labels = ["","",r"Price ($\$$)"],
            legend = legend,  AxesStyle = "Normal")

    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

    gl.savefig(folder_images +'lagsMAs.png', 
               dpi = 100, sizeInches = [16, 9])
               
if (viewing_SEW_windows):
    # Some basic indicators.
    price = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()

    nMA1 = 10
    nMA2 = 20

    SMAw = indl.get_SMA(bMA.delta(nMA1), nMA1, cval = 1)
    EMAw = indl.get_EMA(bMA.delta(nMA1), nMA1, cval = 1)
    WMAw = indl.get_WMA(bMA.delta(nMA1), nMA1, cval = 1)

    SMAw2 = indl.get_SMA(bMA.delta(nMA2), nMA2, cval = 1)
    EMAw2 = indl.get_EMA(bMA.delta(nMA2), nMA2, cval = 1)
    WMAw2 = indl.get_WMA(bMA.delta(nMA2), nMA2, cval = 1)

    SMA = timeData.SMA(n = nMA1)
    
    """ 1st GRAPH """
    ############## Average and Window ################
    ax1 = gl.subplot2grid((1,5), (0,0), rowspan=1, colspan=3)
    title = "Price and SMA. " + str(symbols[0]) + "(" + ul5.period_dic[timeData.period]+ ")"

    gl.plot(dates, [price, SMA] ,AxesStyle = "Normal",
            labels = [title,"",r"Price ($\$$)"],
            legend = ["Price", "SMA(%i)"%nMA1])
    
    ax2 = gl.subplot2grid((1,5), (0,3), rowspan=1, colspan=2)
    gl.stem([], SMAw, nf = 0, AxesStyle = "Normal2",
        labels = ["SMA Window","lag",""],
        legend = ["SMAw(%i)"%nMA1],
        xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
        marker = [".",10,None])
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.35, hspace=0)

    gl.savefig(folder_images +'basicSMA.png', 
               dpi = 100, sizeInches = [14,5])
    """ 2nd GRAPH """
    ############## PLOTTING ################
    gl.set_subplots(3,2)
    
    marker = [".",10,None]
    # Plotting the 3 of them at the same time.
    ax1 = gl.stem([], SMAw, nf = 1,
            labels = ["Windows L = %i"%nMA1,"","SMA"],
            legend = ["SMA(%i)"%nMA1],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = marker, AxesStyle = "Normal2 - No xaxis")
        
    gl.stem([], SMAw2, nf = 1, sharex = ax1, sharey = ax1,
            labels = ["Windows L = %i"%nMA2,"",""],
            legend = ["SMA(%i)"%nMA2], color = "k",
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = marker, AxesStyle = "Normal2 - No yaxis - No xaxis")
    gl.stem([], WMAw, nf = 1, sharex = ax1, sharey = ax1,
            labels = ["","","WMA"],
            legend = ["WMA(%i)"%nMA1],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = marker, AxesStyle = "Normal2 - No xaxis")
    gl.stem([], WMAw2, nf = 1, sharex = ax1, sharey = ax1,
            labels = ["","",""],
            legend = ["WMA(%i)"%nMA2], color = "k",
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = marker, AxesStyle = "Normal2 - No yaxis - No xaxis")
    gl.stem([], EMAw, nf = 1, sharex = ax1, sharey = ax1,
            labels = ["","Lag","EMA"],
            legend = ["EMA(%i)"%nMA1],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = marker, AxesStyle = "Normal2")
    gl.stem([], EMAw2, nf = 1, sharex = ax1, sharey = ax1,
            labels = ["","Lag",""],
            legend = ["EMA(%i)"%nMA2], color = "k",
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = marker, AxesStyle = "Normal2 - No yaxis")
            
    gl.set_zoom(xlim = [-2,nMA2 * (1.10)], ylim = [-0.01, 0.25])
    axes_list = gl.get_axes()
    for ax in axes_list:
        gl.format_yaxis(ax = ax, Nticks = 10)
        
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)

    gl.savefig(folder_images +'windows.png', 
               dpi = 100, sizeInches = [2*8, 2*3])

if (MAMAs):
    # Some basic indicators.
    price = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()

    # For comparing SMA, EMA, WMA
    nHMA = 10
    # Lag of different amplitudes.
    
    delta = bMA.delta(31)
    SMASMA  = indl.get_SMA(delta, nHMA, cval = 1)
    SMASMA =  indl.get_SMA(SMASMA, 2*nHMA, cval = 1)
    
    WMAWMA  = indl.get_WMA(delta, nHMA, cval = 1)
    WMAWMA  = indl.get_WMA(WMAWMA, 2*nHMA, cval = 1)
    
    EMAEMA  = indl.get_EMA(delta, nHMA, cval = 1)
    EMAEMA  = indl.get_EMA(EMAEMA, 2*nHMA, cval = 1)
    
    gl.set_subplots(1,3)
    # Plotting the 3 of them at the same time.
    ax1 = gl.stem([], SMASMA, nf = 1, 
            labels = ["SMASMA","lag","value"],
            legend = ["SMASMA(%i)"%nHMA],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = [".",10,None], AxesStyle = "Normal2")
            
    gl.stem([], WMAWMA, nf = 1,sharex = ax1, sharey = ax1,
            labels = ["WMAWMA","lag"],
            legend = ["WMAWMA(%i)"%nHMA],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = [".",10,None], AxesStyle = "Normal2 - No yaxis")
    gl.stem([], EMAEMA, nf = 1,sharex = ax1, sharey = ax1,
            labels = ["EMAEMA","lag"],
            legend = ["EMAEMA(%i)"%nHMA],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = [".",10,None], AxesStyle = "Normal2 - No yaxis")
            
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.050, hspace=0.01)

    gl.savefig(folder_images +'MAMAw.png', 
               dpi = 100, sizeInches = [2*8, 2*3])
               
if (HullsMA):
    # Some basic indicators.
    price = timeData.get_timeSeries(["Close"]);
    dates = timeData.get_dates()
    
    # For comparing SMA, EMA, WMA
    nHMA = 20
    # Lag of different amplitudes.
    # 
    
    HMA  = indl.get_HMA(price, nHMA)
    WMA  = indl.get_WMA(price, nHMA)
    HMAg  = indl.get_HMAg(price, nHMA)
    
    # For lag and noise

    # Plotting the 3 of them at the same time.
    title = "Hull's MA. " + str(symbols[0]) + "(" + ul5.period_dic[timeData.period]+ ")"

    gl.plot(dates, [price, HMA, HMAg, WMA] , nf = 1 ,AxesStyle = "Normal",
            labels = [title,"",r"Price ($\$$)"],
            legend = ["Price", "HMA(%i)"%nHMA,  "HMAg(%i)"%nHMA, "WMA(%i)"%nHMA])

    gl.savefig(folder_images +'HMA.png', 
               dpi = 100, sizeInches = [2*8, 2*3])

    delta = bMA.delta(25)
    HMAw  = indl.get_HMA(delta, nHMA, cval = 1)
    HMAgw  = indl.get_HMAg(delta, nHMA, cval = 1)
    
    gl.set_subplots(1,2)
    # Plotting the 3 of them at the same time.
    gl.stem([], HMAw, nf = 1,
            labels = ["Hulls Windows","lag","value"],
            legend = ["HMA(%i)"%nHMA],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = [".",10,None], AxesStyle = "Normal2")
    gl.stem([], HMAgw, nf = 1,
            labels = ["Hullsg Windows","lag"],
            legend = ["HMAg(%i)"%nHMA],
            xlimPad = [0.1,0.3], ylimPad = [0.1,0.4],
            marker = [".",10,None], AxesStyle = "Normal2 - No yaxis")
            
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.150, hspace=0.01)

    gl.savefig(folder_images +'HMAw.png', 
               dpi = 100, sizeInches = [2*8, 2*3])
    ############## PLOTTING ################
               