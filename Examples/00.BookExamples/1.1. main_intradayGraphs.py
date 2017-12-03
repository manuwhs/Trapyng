""" BASIC TRADING INDICATORS AND OSCILLATORS PLOTTING"""
# Change main directory to the main folder and import folders
import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Import functions independent of DataStructure
import utilities_lib as ul
import basicMathlib as bMA
import indicators_lib as indl
import indicators_pandas as indp
import oscillators_lib as oscl
import pandas as pd
plt.close("all") # Close all previous Windows

######## SELECT DATASET, SYMBOLS AND PERIODS ########
dataSource =  "GCI"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)

symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
symbols = ["Amazon"]
periods = [5]  # 1440 15
######## SELECT DATE LIMITS ###########
sdate_str = "6-09-2016"
edate_str = "9-09-2016"
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
folder_images = "../pics/gl/"
# Using the library of function built in using the dataFrames in pandas
typeChart = "Bar"  # Line, Bar, CandleStick


# Open and close hours !
opentime = dt.datetime.strptime('2016-09-06 09:30:00', "%Y-%m-%d %H:%M:%S")
closetime = dt.datetime.strptime('2016-09-06 16:00:00',  "%Y-%m-%d %H:%M:%S")
dataTransform = ["intraday", opentime, closetime]

# This is how we would obtain the info if we did not use tradingPlot functions
price = timeData.get_timeSeries(["Close"]);
volume = timeData.get_timeSeries(["Volume"]);
dates = timeData.get_dates()


if (0):
## Try out time mapping func
    caca = ul.transformDatesOpenHours(dates,opentime, closetime )
    semen = ul.detransformDatesOpenHours(caca,opentime, closetime )
    semen = pd.to_datetime(semen)
    for i in range(semen.size):
        diff_sec = (semen[i] - dates[i]).total_seconds()
        print semen[i],  dates[i]
        if (diff_sec != 0):
            print diff_sec/3600
# Get indicators for the price and volume
nMA1 = 15
EMA_price = timeData.EMA(seriesNames = ["Close"], n = nMA1)
SMA_volume = timeData.EMA(seriesNames = ["Volume"], n = nMA1)

ax1 = gl.subplot2grid((4,1), (0,0), rowspan=3, colspan=1)
if (typeChart == "Line"):
    title = "Line Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
    gl.tradingLineChart(timeData,  seriesName = "Close", ax = ax1, 
                legend = ["Close price"],labels = [title,"",r"Price ($\$$)"], 
                AxesStyle = "Normal - No xaxis", dataTransform = dataTransform)
                
elif(typeChart == "Bar"):
     title = "Bar Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
     gl.tradingBarChart(timeData, ax = ax1,  legend = ["Close price"], color = "k",
                        labels = [title,"",r"Price ($\$$)"], AxesStyle = "Normal - No xaxis", 
                        dataTransform = dataTransform)

elif(typeChart == "CandleStick"):
     title = "CandleStick Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
     gl.tradingCandleStickChart(timeData, ax = ax1,  legend = ["Close price"], color = "k",
                        colorup = "r", colordown = "k", alpha = 0.5, lw = 3,
                        labels = [title,"",r"Price ($\$$)"], AxesStyle = "Normal - No xaxis",
                         dataTransform = dataTransform)

#gl.plot(dates, EMA_price, ax = ax1,legend = ["EMA(%i)"%nMA1], 
#            AxesStyle = "Normal")

# ax2: Plot the Volume with the EMA
ax2 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
gl.tradingVolume(timeData, ax = ax2,legend = ["Volume(%i)"%nMA1], 
                 AxesStyle = "Normal", labels = ["","","Volume"],
                 dataTransform = dataTransform)
#gl.plot(dates, SMA_volume, ax = ax2,legend = ["SMA(%i)"%nMA1], 
#        AxesStyle = "Normal - Ny:5")

# Set final properties and save figure
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)

if(type(dataTransform) == type(None)):
    image_name =  "intraDay" + typeChart +'ChartGraph.png'
else:
    image_name = "TransformedX" + typeChart +'ChartGraph.png'
    
gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 6])

