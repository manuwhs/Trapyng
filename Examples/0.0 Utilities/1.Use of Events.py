""" BASIC USAGE OF THE timeData Class AND SOME PLOTTINGS"""
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
import DDBB_lib as DBl
import get_data_lib as gdl
import DDBB_lib as DBl

import eventSystemLib as eSL
plt.close("all") # Close all previous Windows
# Options !
basic_plotting = 1


######## SELECT SOURCE ########
dataSource =  "GCI"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)
folder_images = "../pics/gl/"
######## SELECT SYMBOLS AND PERIODS ########
symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
symbols = ["Amazon", "Alcoa_Inc"]
periods = [15]

######## SELECT DATE LIMITS ###########
sdate = dt.datetime.strptime("21-11-2016", "%d-%m-%Y")
#edate = dt.datetime.strptime("25-11-2016", "%d-%m-%Y")
edate = dt.datetime.now()
######## CREATE THE OBJECT AND LOAD THE DATA ##########
# Tell which company and which period we want
timeData = CTD.CTimeData(symbols[0],periods[0])
TD = DBl.load_TD_from_csv(storage_folder, symbols[1],periods[0])
timeData.set_csv(storage_folder)  # Load the data into the model
timeData.set_TD(TD)

cmplist = DBl.read_NASDAQ_companies(whole_path = "../storage/Google/companylist.csv")
cmplist.sort_values(by = ['MarketCap'], ascending=[0],inplace = True)
symbolIDs = cmplist.iloc[0:30]["Symbol"].tolist()
timeData.set_TD_from_google("AAPL",5,"3d")

#####################################################################3
timeData.set_interval(sdate, edate)
opentime, closetime = timeData.guess_openMarketTime()
period = timeData.guess_period()
print "Period: %f"%period
print "Market Hours " + str(opentime) +" - " + str(closetime)
dataTransform = ["intraday", opentime, closetime]
#dataTransform = None
dataHLOC = timeData.get_timeSeries(["High","Low","Open","Close"])
dates = timeData.get_dates()

gl.barchart(dates, dataHLOC, lw = 2, dataTransform = dataTransform, color = "k",
            AxesStyle = "Normal")


## Define IDs
event_type = "NewAvailableData"
eventID = "AAPL_M5"
handlerID = "handler1"

### Define the event
dataEvent = dict()
dataEvent["symbolID"] = "AAPL"; dataEvent["period"] = 5; dataEvent["src"] = "Google"
myEvent = eSL.CEvent(event_type = event_type, data = dataEvent, eventID = eventID)


## Define the handler 
myHandler = eSL.CHandler(handlerID = "DataDownloader", 
                         handlerFunc = eSL.download_handler_func)

## Define the dispatcher
myDispatcher = eSL.CEventDispatcher()
myDispatcher.add_event_listener(event_type = event_type, handlerObj = myHandler, eventID = None)

myDispatcher.dispatch_event(myEvent)