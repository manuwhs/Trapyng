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
symbolIDs = ["Mad.ITX", "XAUUSD", "XAGUSD", "USA.IBM"]  # Symbols to load
symbolID = "Mad.ITX"  # Symbols to load

mySymbol = CSy.CSymbol(symbolID,periods)  # Set the specified things. Symbol + periods

################ OPERATIONS THAT AFFECT ALL Periods #####################
## Load all the timeSeries and info of the Symbol
mySymbol.set_csv(storage_folder)
mySymbol.load_info(info_folder)
print mySymbol.info  # print information about the symbol

# timeData operations
myTimeData = mySymbol.get_timeData(period = 5) # Obtain one of the timeDatas
mySymbol.del_timeData(period = 15)             # Remove one of the periods
periods_Symbol = mySymbol.get_periods()  # [1440, 60, 5] Not ordered

timeData_15 = CTD.CTimeData(symbolID,15) # Create an independent timeData object
timeData_15.set_csv(storage_folder)      # Load the data into the model
mySymbol.add_timeData(period = 15, timeDataObj = timeData_15) # Add the timeData 


# Set time limits to all the TD of the object
sdate_str = "03-09-2016"; edate_str = "10-09-2016"
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
periods = sorted(periods,reverse = True)

opentime, closetime = mySymbol.get_timeData(periods[1]).guess_openMarketTime()

dataTransform = ["intraday", opentime, closetime]
folder_images = "../pics/gl/"
gl.set_subplots(4,1)
 # TODO: Be able to automatize the shareX thing
axeshare = None
for i in range(len(periods)):
    period = periods[i]
    myTimeData = mySymbol.get_timeData(period)
    
    AxesStyle = " - No xaxis"
    if (i == len(periods) -1):
        AxesStyle = ""
    
    if (i == 0):
        title = "Bar Chart. " + str(symbols[0]) + r" . Price ($\$$)"
    else:
        title = ""
    ylabel =   ul.period_dic[myTimeData.period]
    ax = gl.tradingBarChart(myTimeData,  legend = ["Close price"], color = "k", 
                nf = 1, sharex = axeshare, labels = [title,"",ylabel], AxesStyle = "Normal" + AxesStyle, 
                dataTransform = dataTransform)
    axeshare = ax
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)
image_name = "differentPeriods.png"
gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [30, 12])
# TODO: We can also use the library of indicators especifing the period. 
# If no period specified, we use the dayly or the bigges one.


#######################
