""" BASIC USAGE OF THE CLASS SYMBOL"""
# Change main directory to the main folder and import folders
### THIS CLASS DOES NOT DO MUCH, A QUICK WAY TO AFFECT ALL THE TIMESERIES
## OF THE SYMBOL IN A CERTAIN MANNER, like loading, or interval.
## It also contains info about the Symbol Itself that might be relevant.

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
import CSymbol as CSy
# Import functions independent of DataStructure
import utilities_lib as ul
plt.close("all") # Close all previous Windows
######## SELECT DATASET, SYMBOLS AND PERIODS ########
dataSource =  "Hanseatic"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)
#### Load the info about the available symbols #####
Symbol_info = CSy.load_symbols_info(info_folder)
Symbol_names = Symbol_info["Symbol_name"].tolist()
Nsym = len(Symbol_names)

############################### OUR SYMBOL ############################
periods = [5,15,60,1440]   # Periods to load
symbols = ["Mad.ITX", "XAUUSD", "XAGUSD", "USA.IBM"]  # Symbols to load
mySymbol = CSy.CSymbol(symbols[0],periods)  # Set the specified things. Symbol + periods

################ OPERATIONS THAT AFFECT ALL Periods #####################
## Load all the timeSeries and info of the Symbol
mySymbol.set_csv(storage_folder)
mySymbol.load_info(info_folder)
print mySymbol.info  # print information about the symbol

# Set time limits to all the TD of the object
sdate_str = "01-04-2016"; edate_str = "01-05-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
mySymbol.set_interval(sdate,edate) # Set the interval period to be analysed

#### Filling of data !!!
#mySymbol.fill_data()

# Set the timeSeries to operate with.
mySymbol.set_seriesNames(["Close"])

############################################################
###### NOW WE OPERATE DIRECTLY ON THE TIMEDATAs ############
############################################################

periods = mySymbol.get_periods()

gl.set_subplots(4,1, sharex = True)
# TODO: Be able to automatize the shareX thing
for period in periods:
    myTimeData = mySymbol.get_timeDataObject(period)
    price = myTimeData.get_timeSeries(["Close"])
    volume = myTimeData.get_timeSeries(["Volume"])
    dates = myTimeData.get_dates()
    
    gl.plot(dates, price, labels = ["", "", mySymbol.symbol],
            legend = [str(period)])
            
    gl.stem(dates, volume,
            legend = [str(period)], na = 1, nf = 0)
            
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

# TODO: We can also use the library of indicators especifing the period. 
# If no period specified, we use the dayly or the bigges one.


#######################
