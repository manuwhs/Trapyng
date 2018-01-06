""" EXPERIMENTAL INDICATORS """
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
sdate_str = "01-01-2016"; edate_str = "2-1-2017"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Tell which company and which period we want
timeData = CTD.CTimeData(symbols[0],periods[0])
timeData.set_csv(storage_folder)  # Load the data into the model
timeData.set_interval(sdate,edate) # Set the interval period to be analysed

###########################################################################
############## TIME SERIES INDICATORS #####################################
###########################################################################

advanced_smoothing_f = 0
if (advanced_smoothing_f == 1):
    price = timeData.get_timeSeries(["Average"]);
    casd = ul.get_Elliot_Trends(price,10);
    timeData.plot_timeSeries()
    flag_p = 0
    for trend in casd:
        gl.plot(timeData.dates[trend], price[trend], lw = 5, nf = flag_p)
        flag_p = 0
        
###########################################################################
############## Trials ############################################
###########################################################################
try_ind_f = 0
if (try_ind_f == 1):
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.dates
    a,b = indl.MDD(price, 100)
    
    gl.plot(dates, price, labels = ["Maximum DrawDown","", "Price"],
            legend = ["Price"])
    gl.plot(dates, [a,b], nf = 0, na = 1,
            labels = ["MDD", "", "MMDd"],
            legend = ["Dayly MDD", "Window MDD"])
    
    ## Relative Strength Index !
    RSI = timeData.get_RSI(N = 14)
    RSI2 = oscl.get_RSI2(price, n = 14)
    gl.plot(dates,price, legend =  ["Price"], 
            nf = 1, na = 0)
            
    gl.plot(dates,RSI, legend =  ["RSI"], 
            nf = 0, na = 1)
            
    gl.plot(dates,RSI2, legend =  ["RSI2"], 
            nf = 0, na = 0)  

###########################################################################
############## Pandas indicator Library ############################################
###########################################################################
pandas_lib = 0
if (pandas_lib == 1):
    
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()
    df = timeData.get_timeData()
    
###########################################################################
    # Simple Moving Average
    SMA = timeData.SMA(seriesNames = ["Close"], n = 10)
    EMA = timeData.EMA(seriesNames = ["Close"], n = 10)
 
    gl.plot(dates, [price, SMA, EMA] , nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price", "SMA", "EMA"])
            
###########################################################################
    # Bollinger Bands, Pivot points Resistences and Supports and ATR
    BB = timeData.BBANDS(seriesNames = ["Close"], n = 10)
    ATR = timeData.ATR(n = 20)
    PPSR = timeData.PPSR()
    
    gl.set_subplots(3,1)
    gl.plot(dates, [price, BB[:,0],BB[:,1]] , nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price", "Bollinger Bands"])
            
    gl.fill_between(x = dates, y1 = BB[:,0], y2 = BB[:,1], alpha = 0.5)
    
    gl.plot(dates, price, nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price"])
    gl.plot(dates,  PPSR , nf = 0,
            legend = [ "Supports and Resistances"])
 
    gl.plot(dates, price , nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price"])
    gl.plot(dates, ATR , nf = 0, na = 1,
            labels = ["Averages","Time","Value"],
            legend = ["ATR"], fill = 1)
            
pandas_lib1 = 0
if (pandas_lib1 == 1):
###########################################################################
# Momentum and Rate of convergence
    MOM = timeData.MOM(n = 1)
    ROC = timeData.ROC(n = 2)
    gl.plot(dates, ROC, na = 1, nf = 0,
            legend = ["ROC"])
    
    gl.plot(dates, price , nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price", " Momentum", "ROC"])
    gl.plot(dates, MOM , nf = 0, na = 1,
            legend = ["Momentum"])
    gl.plot(dates, ROC , nf = 0, na = 1,
            legend = ["ROC"])

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
    gl.plot(dates, ADX , nf = 0, na = 0,
            legend = ["ADX"])
            
    gl.plot(dates, ACCDIST , nf = 1, na = 0,
            legend = ["ACCDIST"])
#    ws = 40
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
    
pandas_lib2 = 0
if (pandas_lib2 == 1):

    MACD, MACDsign, MACDdiff = timeData.MACD()
    TRIX = timeData.TRIX()
    
    gl.set_subplots(3,1)
    

    gl.plot(dates, price , nf = 1,
            labels = ["Averages","Time","Value"],
            legend = ["Price"])
            
    gl.plot(dates, [MACD, MACDsign, MACDdiff], nf = 1, na = 0,
            legend = ["MACD"])
    gl.plot(dates,  TRIX , nf = 1, na = 0,
            legend = [ "TRIX"])
            
    gl.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
    gl.create_axes(position = [0.3,0.4,0.3,0.4])
    
#    gl.subplot2grid((10,10),)
#    gl.subplot2grid((6,4), (5,0), ro)
    gl.plot(dates,  TRIX , nf = 0, na = 0,legend = [ "TRIX"])


#    gl.plot(dates, ADX , nf = 1, na = 0,
#            legend = ["ADX"])
#            
#    gl.plot(dates, ACCDIST , nf = 1, na = 0,
#            legend = ["ACCDIST"])       
#    
###########################################################################
############## Moving Averages ############################################
###########################################################################

MA_f = 0
if (MA_f == 1):
    ############################################
    ## Plotting of the 3 basic moving averages
    price = timeData.get_timeSeries(["Close"]);
    # It also works for compound signals
#    price = timeData.get_timeSeries(["Close","Average"]);
    SMATD = timeData.get_SMA(10)
    WMATD = timeData.get_WMA(10)
    EMATD = timeData.get_EMA(10)
    
    # Actual plotting
    gl.plot(timeData.dates, price, nf = 1, 
            labels = ["3 basic Moving Averages","Time","Value"],
            legend = ["Price"])
            
    gl.plot(timeData.dates, SMATD, nf = 0,
            legend = ["Simple"])
    gl.plot(timeData.dates, WMATD, nf = 0,
            legend = ["Weighted"])
    gl.plot(timeData.dates, EMATD, nf = 0,
            legend = ["Exponential"])
            
    ############################################
    ## Plotting of the Triple Cross of Death
    TrCrMr =  timeData.get_TrCrMr()
    timeData.plot_TrCrMr()
    
    ############################################
    ## Hulls moving averages
    HMAg =  timeData.get_HMAg(L = 30)
    # Actual plotting
    gl.plot(timeData.dates, price, nf = 1, 
            labels = ["Generalized Hull MA","Time","Value"],
            legend = ["Price"])
    gl.plot(timeData.dates, SMATD, nf = 0,
            legend = ["Generalized Hull MA"])

    ############################################
    ## Plot the Trained Mean
    TMA =  timeData.get_TMA(L = 30)
    # Actual plotting
    gl.plot(timeData.dates, price, nf = 1, 
            labels = ["Generalized Hull MA","Time","Value"],
            legend = ["Price"])
    gl.plot(timeData.dates, TMA, nf = 0,
            legend = ["Trained Mean"])
            
#############################################################
############## Oscillators #################################
#############################################################
oscil_f = 0
if (oscil_f == 1):
#    gl.set_subplots(2,2)
    timeSeries = timeData.get_timeSeries(["Average"]);
    dates = timeData.dates
#    dates = range(len(dates))
    ## Plot the momentum
    momentum = timeData.get_momentum(N = 10)
    RSI = timeData.get_RSI(N = 5)
#    timeData.plot_timeSeries()
#    gl.plot(dates,momentum, legend =  ["momentum"], 
#            nf = 0, na = 1,
#            position = [0.05,0.05,0.9,0.3])
#    
#    ws = 40
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
    
    # Plot the MACD 
    mshort, mlong, MACD = timeData.get_MACD()
    gl.plot(dates,[timeSeries, mshort, mlong], legend = [timeData.seriesNames, "fast","slow"])

    low_ax_pos = [0.05,0.05,0.9,0.35]
    low_ax_pos2 = [0.05,0.05,0.9,0.3501]    
    low_ax_pos3 = [0.05,0.05,0.9,0.3502]    
    gl.plot(dates,RSI, legend =  ["RSI"], 
            nf = 0, na = 1,
            position = low_ax_pos)
            
    gl.plot(dates,momentum, legend =  ["momentum"], 
            nf = 0, na = 1,
            position = low_ax_pos3)
#            
    gl.plot(dates,mshort, legend =  ["MACD"], nf = 0, na = 1,
            position =low_ax_pos2)
            
    ws = 200
    gl.add_slider(args = {"wsize":ws}, plots_affected = [])


#############################################################
############## Volatility #################################
############################################################
vol_f = 0
if (vol_f == 1):
    timeSeries = timeData.get_timeSeries(["Average"]);
    BB = timeData.get_BollingerBand(5)
    timeData.plot_BollingerBands(5)
    
    ATR = timeData.get_ATR()
    timeData.plot_timeSeries()
    gl.plot(timeData.dates,ATR, legend =  ["MACD"], nf = 0, na = 1)

#############################################################
##############  Velero Graphs #################################
############################################################
velero_f = 0
if (velero_f == 1):
    ws = 100
#    timeSeries = timeData.get_timeSeries(["Average"]);
#    gl.plot(timeData.dates, timeSeries)
   
    gl.Velero_graph(timeData.TD.ix[timeData.time_mask], ws = ws, nf = 1)
    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
    gl.add_hidebox()
    
#    list_values = []
#    gl.add_selector(list_values)
#    gl.add_onKeyPress()

#    gl.Heiken_Ashi_graph(timeData.TD.ix[timeData.time_mask], nf = 0)
    
#############################################################
##############  Random graphical Properties #################################
#############################################################

randomp_f = 0
if (randomp_f == 1):
    gl.set_subplots(2,2)
    timeData.scatter_deltaDailyMagic()

trading_station = 0
if (trading_station == 1):
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
    
#############################################################
############## Data filling  #################################
#############################################################

data_filling = 0
if (data_filling == 1):
    time_diff = intl.find_min_timediff(timeData.TD)
#    print time_diff
#    print type(time_diff)
    
#    idx = intl.get_pdintra_by_days(timeData.TD)
    
    ## Fill the interspaces, create another timeSeries and plot it
    filled_all = intl.fill_everything(timeData.get_timeData())
    
    timeData2 = copy.deepcopy(timeData)
    timeData2.set_timeData(filled_all)
    timeData2.get_timeSeries(["Close"])
    timeData2.plot_timeSeries()
    print timeData2.get_timeSeries().shape
    
    ## Fill missing values by first filling everythin
    filled = intl.fill_by_filling_everything(timeData.get_timeData())
    timeData2 = copy.deepcopy(timeData)
    timeData2.set_timeData(filled)
    timeData2.get_timeSeries(["Close"])
    timeData2.plot_timeSeries(nf = 0)
    print timeData2.get_timeSeries().shape
    
    ### Get the day table
    pd_dayly = intl.get_dayCompleteTable(timeData.get_timeData())

    time_index = intl.find_trade_time_index(timeData.get_timeData())

    index_shit = intl.find_interval_date_index(timeData.get_timeData(), dt.date(2016,3,1),  dt.date(2016,5,1))

#############################################################
############## Widgets  #################################
#############################################################
def convert_dates_str(X):
    # We want to convert the dates into an array of char so that we can plot 
    # this shit better, and continuous

    Xdates_str = []
    for date_i in X:
        name = date_i.strftime("%Y/%m/%d:%H:%M")
        Xdates_str.append(name)
    return Xdates_str
    
widgets_flag = 0
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
    
    