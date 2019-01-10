"""
This file contains the initial code to load the data, explore its format,
visualize it and decide on the preprocessing.
"""
# %% 
# Load all the directories needed for the code to be executed both from
# a console and Spyder, and from the main directory and the local one.
# Ideally this code should be executed using Spyder with working directory

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Specific utilities
import utilities_lib as ul
import toptal_utils as tut

plt.close("all") # Close all previous Windows

# %% 
"""
################### EXECUTING OPTIONS ###################
"""

folder_images = "../pics/Toptal/"
storage_folder = ".././storage/Toptal/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

# Using the library of function built in using the dataFrames in pandas
typeChart = "Bar"  # Line, Bar, CandleStick
# Remove the intraday gaps of out of session
tranformIntraday = 0
# Move the start of the trading session to the start of the natural day
adjust_session_start = 1
# Remove the non-trading days
remove_nontrading_days = 1
# Remove days with little amount of trading samples
remove_small_trading_days = 1
# Divide the data in daily trading sessions.
get_daily_data = 0
# Plot the data 
plotting_flag = 1
# Plots of the daily information
plotting_daily = 1
######## SELECT ASSET AND PERIOD ###########
symbols = ["EURUSD"]
periods = [15]  # 1440 15

######## SELECT DATE LIMITS ###########
## We set one or other as a function of the timeSpan
sdate_str = "01-01-2010"; sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate_str = "31-12-2016"; edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
edate_str = "28-02-2010"; edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
#edate_str = "28-02-2012"; edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

#sdate_str = "15-03-2010"; sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
#edate_str = "10-04-2010"; edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

#sdate_str = "21-10-2010"; sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
#edate_str = "13-11-2010"; edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

# %% 
"""
################### LOAD THE DATA OPTIONS ###################
"""

######## CREATE THE DATASTRUCTURE OBJECT AND LOAD THE DATA ##########
# Speify which company and which period we are loading
timeData = CTD.CTimeData(symbols[0],periods[0])
timeData.set_csv(storage_folder,file_name)  # Load the data into the model
timeData.set_interval(sdate,edate, trim = True) 
 
#print ("timeData shape: ", timeData.TD.shape)
if (remove_nontrading_days):
#    timeData.TD.index += pd.Timedelta(hours = 2)

    ################# Remove all non-trading samples #######################
    timeData.TD = timeData.TD[timeData.TD["Volume"] > 0]
    timeData.TD.reset_index()
    # Reset the index values of the table and call set_interval to reset the internal mask
    timeData.set_interval(sdate,edate) 

## Readjust time so that the session starts at 00:00 and not at 22:00-20:00 of the
# previous natural day 
if (adjust_session_start):
    ## Get the begginings of the starting sessions !!
    week_start_times = np.where(timeData.TD.index[1:]-timeData.TD.index[:-1] > dt.timedelta(minutes = 15)) [0] 
    week_start_times += 1  # We shift all the postiions one to get the start of the week
    
    ############## PRINT THE INFORMATION ABOUT TRADING SESSIONS #########
    for i in range(week_start_times.size -1):
        duration = timeData.TD.index[week_start_times[i+1]-1] - \
        timeData.TD.index[week_start_times[i]] + dt.timedelta(minutes = 15)
        closed_time = timeData.TD.index[week_start_times[i+1]] - \
        timeData.TD.index[week_start_times[i]] - duration
        print (" Session: ",timeData.TD.index[week_start_times[i]].dayofweek, " ", 
               timeData.TD.index[week_start_times[i]]," Duration: ", duration, " Then closed: ", closed_time)
    ## Last session   
    duration = timeData.TD.index[-1] -  timeData.TD.index[week_start_times[-1]]  + dt.timedelta(minutes = 15)
    print ("Continuous session: ",timeData.TD.index[week_start_times[-1]]," Duration: ", duration)
    
    ################################################
    ######  Readjust time by aligning each init of trading session to 00:00:00
    ######################################################
    
    index_copy = copy.deepcopy(timeData.TD.index)
    timeData.TD['Original Time'] = pd.Series(index_copy, index=timeData.TD.index)
    
    ## The first week we assume the starting session is at 22:00
    start_hour = 22
    timeData.TD.index.values[:week_start_times[0]] =  timeData.TD.index[:week_start_times[0]] + \
    dt.timedelta(hours =   24 - start_hour)
    
    for i in range(week_start_times.size -1):
        start_hour = timeData.TD.index[week_start_times[i]].hour
        if (start_hour > 0):
            # We cannot change the index, but we can change the values
            timeData.TD.index.values[week_start_times[i]:week_start_times[i+1]] =  \
            timeData.TD.index[week_start_times[i]:week_start_times[i+1]] + dt.timedelta(hours =   24 - start_hour)
    ## The last week:
    start_hour = timeData.TD.index[week_start_times[-1]].hour
    if (start_hour > 0):
        # We cannot change the index, but we can change the values
        timeData.TD.index.values[week_start_times[-1]:] =  timeData.TD.index[week_start_times[-1]:] + \
        dt.timedelta(hours =   24 - start_hour)
    
    ## Rest interval after we changed the index
    timeData.set_interval(sdate,edate) 
    
if (get_daily_data):
    ## Get the valid trading days sorted
    ## Each day is suposed to have 96 trading 15 min slots 
    days_keys, day_dict = timeData.get_indexDictByDay()
    print ("Number of initial days: %i"%(len(days_keys)))
    
    ############ Print the Irregular days ##############
    for day in days_keys:
        if (len(day_dict[day]) != 96):
            print (day, len(day_dict[day]))
    
    print ("Number of trading days %i"%(len(days_keys)))
    if (remove_small_trading_days):
        for day_i in range(len(days_keys)):  
            # We go from end to begining
            day_index = len(days_keys) -1 - day_i
            day = days_keys[day_index]
            if (len(day_dict[day]) <= 80):
                print ("Removed day: ", day, " Samples: ", len(day_dict[day]))
                timeData.TD.drop(timeData.TD.index[day_dict[days_keys[day_index]]], inplace=True)
    #            print ("timeData shape: ", timeData.TD.shape)
                
        # Reset the index of the pd Datagrame and the timeData structure
        timeData.TD.reset_index()
        timeData.set_interval(sdate,edate) 
                    
    days_keys, day_dict = timeData.get_indexDictByDay()
    print ("Number of final trading session: %i"%(len(days_keys)))
    
# %% 
"""
##########################################################
############### PLOTTING #################################
##########################################################
"""

if (plotting_daily):
    
    # Get the daily values of the remaining days
    timeData_daily = tut.get_daily_timedata(timeData, symbols[0])
    ## Get the daily HLOC
    H,L,O,C,V = np.array(timeData_daily.TD[["High","Low","Open","Close","Volume"]][:]).T
    
    gl.init_figure()
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=3, colspan=1)
    
    title = "Bar Chart. " + str(symbols[0]) + "(" + ul.period_dic[1440]+ ")" 
    gl.tradingBarChart(timeData_daily, ax = ax1,  legend = ["Close price"], color = "k",
                        labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis")
         
    ax2 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    gl.tradingVolume(timeData_daily, ax = ax2,legend = ["Volume"], 
                     AxesStyle = "Normal", labels = ["","","Volume"], color = "#53868B")
    
    
    gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 10, xticks = 10, yticks = 10)
    image_name = "daily_data"
    gl.savefig(folder_images + image_name, 
               dpi = 100, sizeInches = [20, 7])
    
if (plotting_flag):
    # Open and close hours !
    if(tranformIntraday):
        opentime, closetime = timeData.guess_openMarketTime()
        dataTransform = ["intraday", opentime, closetime]
    else:
        dataTransform = None 

    # This is how we would obtain the info if we did not use tradingPlot functions
    price = timeData.get_timeSeries(["Close"]);
    volume = timeData.get_timeSeries(["Volume"]);
    dates = timeData.get_dates()
    
    gl.init_figure()
    # Get indicators for the price and volume
    nMA1 = 30
    EMA_price = timeData.EMA(seriesNames = ["Close"], n = nMA1)
    SMA_volume = timeData.EMA(seriesNames = ["Volume"], n = nMA1)
    
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=3, colspan=1)
    
    if (get_daily_data):
        gl.step(days_keys, [H,L,C], legend = ["H","L","C"], lw = 3, ax = ax1, 
                where = "post",  dataTransform = dataTransform)
    
    if (typeChart == "Line"):
        title = "Line Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
        gl.tradingLineChart(timeData,  seriesName = "Close", ax = ax1, 
                    legend = ["Close price"],labels = [title,"",r"Rate"], 
                    AxesStyle = "Normal - No xaxis", dataTransform = dataTransform)
                    
    elif(typeChart == "Bar"):
         title = "Bar Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
         gl.tradingBarChart(timeData, ax = ax1,  legend = ["Rate"], color = "k",
                            labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis", 
                            dataTransform = dataTransform)
    
    elif(typeChart == "CandleStick"):
         title = "CandleStick Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
         gl.tradingCandleStickChart(timeData, ax = ax1,  legend = ["Rate"], color = "k",
                            colorup = "r", colordown = "k", alpha = 0.5, lw = 3,
                            labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis",
                             dataTransform = dataTransform)
    
    gl.plot(dates, EMA_price, ax = ax1,legend = ["EMA(%i)"%nMA1], dataTransform = dataTransform,
                AxesStyle = "Normal", color = "b")
    
    # ax2: Plot the Volume with the EMA
    ax2 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    gl.tradingVolume(timeData, ax = ax2,legend = ["Volume"], 
                     AxesStyle = "Normal", labels = ["","","Volume"],
                     dataTransform = dataTransform, color = "#53868B")
    
    gl.plot(dates, SMA_volume, ax = ax2,legend = ["SMA(%i)"%nMA1], dataTransform = dataTransform,
            AxesStyle = "Normal - Ny:5")
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    
    gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    
    if(type(dataTransform) == type(None)):
        image_name =  "intraDay" + typeChart +'ChartGraph.png'
    else:
        image_name = "TransformedX" + typeChart +'ChartGraph.png'
        
    gl.savefig(folder_images + image_name, 
               dpi = 100, sizeInches = [20, 7])


    
