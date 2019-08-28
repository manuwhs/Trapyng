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
################# STORAGE FUNCTIONS ##############
storage_f = 0
if (storage_f == 1):
    timeData.add_csv(updates_folder) # Add more data to the tables
    timeData.save_to_csv(storage_folder)
################# INTERVAL PROPERTIES ##############
interval_f = 0
if (interval_f == 1):
    print timeData.dates.shape  ## The dates of the selected data
    print timeData.time_mask.shape  ## Array of indexes of the selected dates
    # It is a numpy array

#############################################################
############## Obtain time series #################################
#############################################################
price = timeData.get_timeSeries(["Open","Close","Average"]);
dates = timeData.get_dates()

###########################################################################
############## Dayly things obtaining  ####################################
###########################################################################

## We have to fix the fact of TimeZone, the fact that on
## Sundays we have only one hour 23:00 - 24:00 or something like that
## Maybe displaze date and join stuff.

dayly_data_f = 0
if (dayly_data_f == 1):
    prices_day, dates_day = timeData.get_intra_by_days()
    
    plot_flag = 1
    for day in range (len(prices_day)):
        # type(dates_day[1]) <class 'pandas.tseries.index.DatetimeIndex'>
        # I could not find a fucking way to modify it inline TODO
        # This is done this way becasue we cannot asume that we have all the data
        # for all the days
        list_time = []
        for i in range(len(dates_day[day])):
            # datetime.replace([year[, month[, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]]]]])
            dateless = dates_day[day][i].replace(year = 2000, month = 1, day = 1)
            list_time.append(dateless)
        gl.plot(list_time, prices_day[day],
                nf = plot_flag)
        plot_flag = 0

###########################################################################
############## Trend Detection ############################################
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
try_ind_f = 1
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
pandas_lib = 1
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
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()
    df = timeData.get_timeData()

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
    gl.plot(dates, ADX , nf = 0, na = 1,
            legend = ["ADX"])
            
    gl.plot(dates, ACCDIST , nf = 1, na = 0,
            legend = ["ACCDIST"])
#    ws = 40
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
    
pandas_lib2 = 0
if (pandas_lib2 == 1):
    price = timeData.get_timeSeries(["Average"]);
    dates = timeData.get_dates()
    df = timeData.get_timeData()
    
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

trading_indicators_complex = 0
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
##############  Random graphical Properties #################################
#############################################################

randomp_f = 0
if (randomp_f == 1):
    gl.set_subplots(2,2)
    timeData.scatter_deltaDailyMagic()

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