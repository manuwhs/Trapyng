""" 
BASIC USAGE OF THE timeData Class AND SOME PLOTTINGS
In this file you 


"""
# Change main directory to the main folder and import folders with own libraries
import os
os.chdir("../../")
import import_folders 

####### Public Libraries ######
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

####### Own Libraries ######
from graph_lib import gl  # Own graphical library
import CTimeData as CTD  # Data Structures for basic timeseries
# Import functions independent of DataStructure
import utilities_MQL5 as ul5 # General utilities related to MQL5
import DDBB_lib as DBl      # Database library
plt.close("all") # Close all previous Windows


############ print options ############
storage_f = 0           # Print 
show_data_shape = 0
guessing_markerhours = 0
basic_timeSeries_functions = 0

################# Plotting  Options! ################
basic_plotting = 0  # Basic chart with signal, volume, return
intrabydays_f = 0   # We divide the intraday data into its dayly components and plot all days on top of each other

################ SELECT SOURCE AND FOLDER NAMES ################
dataSource =  "MQL5"  # Hanseatic  FxPro GCI Yahoo
[storage_folder,  updates_folder] = ul5.get_foldersData(source = dataSource)
folder_images = "../pics/gl/"

######## SELECT SYMBOLS AND PERIODS ########
symbols = ["EURUSD", "Alcoa_Inc"]
periods = [15]

######## SELECT DATE LIMITS ###########
sdate = dt.datetime.strptime("15-8-2018", "%d-%m-%Y")
edate = dt.datetime.strptime("25-8-2018", "%d-%m-%Y")

######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Create the information object for the timeSeries
timeData = CTD.CTimeData(symbols[0],periods[0])

# Create the pandas dataframe with the OHCLV timeSeries and add it to the timeData
# Both approaches do the same
if(1):
    TD = DBl.load_TD_from_csv(storage_folder, symbols[0],periods[0])
    timeData.set_TD(TD)
else:
    timeData.set_csv(storage_folder)  # Load the data into the model

# Set the interval period to be analysed
timeData.set_interval(sdate,edate, trim = True) 

# In order to plot intraday data without discontnuities we need to know 
# its market hours.
opentime, closetime = timeData.guess_openMarketTime()
period = timeData.guess_period() # Not needed but implemented
# Information needed for proper plotting.
dataTransform = ["intraday", opentime, closetime]

print ("Loaded: %s %i minutes"%(timeData.symbolID, timeData.period))
print ("Market Hours " + str(opentime) +" - " + str(closetime))
print ("Loaded dates: ", sdate," to ", edate)

################# STORAGE FUNCTIONS ##############
## Add more candlesticks from the MQL trading 
if (storage_f == 1):
    timeData.add_csv(updates_folder) # Add more data to the tables
    timeData.save_to_csv(storage_folder)

############## Data filling  #################################
# TODO: call data filling functions

############## OBTAIN TIME SERIES #################################
# We can obtain some basic preprocessed time series
price = timeData.get_timeSeries(["Open","Close","Average"]);
dates = timeData.get_dates()  ## The dates of the selected data

if (show_data_shape):
    """
    Show the shape of different data structures
    """
    print (price.shape) # np.array(Nsamples, Nseries)
    print (dates.shape) # np.array(Nsamples, ) of selected Dates
    print (timeData.time_mask.shape) # # np.array(Nsamples, ) of indices of dates

############## OWN BASIC PLOTING FUNC #########################################
if (basic_timeSeries_functions):
    """
    Functions to obtain the basic timeSeries data from the object.
    """
    # If we do not specify new time series, we use the last ones, default = close.
    price = timeData.get_timeSeries(["RangeHL","RangeCO"]);
    price = timeData.get_timeSeries(["magicDelta"]);

    returns = timeData.get_timeSeriesReturn(["Close","Average"])
    cumReturns = timeData.get_timeSeriesCumReturn()
    SortinoRatio = timeData.get_SortinoR()
    get_SharpR = timeData.get_SharpR()
    

################# BASIC PLOTTING MIXTURES ############################
if (basic_plotting):
    """ 
    We aim to plot the price, volume, return and cummulative return for the selected security 
    in the selected time frame.
    """
    gl.set_subplots(4,1)
    ############# 1: Basic Time Series and Volume
    seriesNames = ["Average", "High"]
    dataHLOC = timeData.get_timeSeries(["High","Low","Open","Close"])
    prices = timeData.get_timeSeries(seriesNames);
    dates = timeData.get_dates()
    volume = timeData.get_timeSeries(["Volume"]);
    Returns = timeData.get_timeSeriesReturn(seriesNames = ["Close"]);
    CumReturns = timeData.get_timeSeriesCumReturn(seriesNames = ["Close"]);
    nSMA = 10
    SMA = timeData.SMA(n = nSMA)
    
    ax1 = gl.plot(dates, SMA, labels = [timeData.symbolID + str(timeData.period), "Time", "Price"],
            legend = ["SMA(%i)" % nSMA], nf = 1, dataTransform = dataTransform,
            AxesStyle = "Normal - No xaxis", color = "cobalt blue") 
    
    gl.barchart(dates, dataHLOC, lw = 2, dataTransform = dataTransform, color = "k",
                AxesStyle = "Normal - No xaxis", legend = ["OHCL"])
    # Notice the Weekends and the displacement between bars and step
    
    ############# 2: Volume
    gl.stem(dates, volume, sharex = ax1, labels = ["", "Time", "Volume"],
            legend = ["Volume"], nf = 1, alpha = 0.8, dataTransform = dataTransform,
            AxesStyle = "Normal - No xaxis")

    ############# 3: Returns
    gl.stem(dates, Returns, sharex = ax1, labels = ["", "Time", "Return"],
           legend = ["Return"], nf = 1, dataTransform = dataTransform,
           AxesStyle = "Normal - No xaxis")

    ############# 4: Commulative Returns 
    seriesNames = ["Close"]

    gl.plot(dates, CumReturns, sharex = ax1,labels = ["", "Time", "Cum Return"],
           legend =  ["Cum Return"], nf = 1, dataTransform = dataTransform,
            AxesStyle = "Normal")
            ############## Dayly things obtaining  ####################################
# We separate the data into a list of days in order to be able to analyze it easier

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

    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)
    image_name = "timeDataExample.png"
    gl.savefig(folder_images + image_name, 
               dpi = 100, sizeInches = [30, 12])
    # TODO: Plot reconstruction to show that it is not the same.


if(intrabydays_f):
    """ 
    We aim to plot the intraday price and volume for different days on top of each other.
    For this purpose, we first divide the entire timeseries into its corresponding days and
    then we plot day by day
    """
    
    TD = timeData.get_TD()
    #caca = TD.groupby(TD.index.map(lambda x: x.date))
    caca = TD.groupby(TD.index.date)
    groups_of_index_dict = caca.groups # This is a dictionary with the dates as keys and the indexes of the TD as values
    days_dict = caca.indices # This is a dictionary with the dates as keys and the indexes of the TD as valu
    keys = days_dict.keys()# list of datetime.date objects
    keys = sorted(keys)
    set_indexes = days_dict[keys[0]]
    
    gl.set_subplots(2,1)
    ax1 = gl.plot([],[], nf = 1)
    ax2 = gl.plot([],[], nf = 1, sharex = ax1)
    
    for key in keys:
        set_indexes = days_dict[key]
        set_indexes.sort()
        set_indexes = timeData.time_mask[set_indexes] # Since it changed.
        
        times = timeData.get_dates(set_indexes).time
        values = timeData.get_timeSeriesCumReturn(["Close"],set_indexes)
        
        gl.plot(times, values, ax = ax1)
        volume = timeData.get_timeSeries(["Volume"],set_indexes)
        gl.scatter(times, volume, ax = ax2)

    
