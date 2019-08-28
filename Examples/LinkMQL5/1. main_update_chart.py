"""
This file is meant to be a first attempt to connect to MT5 and get 
the new candlesticks and update the chart !
"""

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
import CSymbol as CSy
# Import functions independent of DataStructure
import utilities_MQL5 as ul5 # General utilities related to MQL5
plt.close("all") # Close all previous Windows

import sockets_lib, numpy as np
# Now we can proceed to creating a class responsible for socket manipulation:
plt.close("all")
"""
####################### OPTIONS ###########################
"""
folder_images = "../pics/gl/"
dataSource =  "MQL5"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, updates_folder] = ul5.get_foldersData(source = dataSource)
#### Load the info about the available symbols #####
Symbol_info = CSy.load_symbols_info(storage_folder)

############################### OUR SYMBOL ############################
periods = [5]   # Periods to load
symbolID = "EURUSD"  # Symbols to load
n_days_show = 3
sdate = dt.datetime.now() - dt.timedelta(days=n_days_show)
edate = dt.datetime.now()


mySymbol = CSy.CSymbol(symbolID,periods)  # Set the specified things. Symbol + periods
mySymbol.set_csv(storage_folder)
mySymbol.set_interval(sdate, edate)

myTimeData = mySymbol.get_timeData(periods[0])
opentime, closetime = myTimeData.guess_openMarketTime()


def redraw_chart(myTimeData):
    
    dataTransform = ["intraday", opentime, closetime]
    period = periods[0]
    myTimeData = mySymbol.get_timeData(period)
    AxesStyle = ""
    title = "Bar Chart. " + str(symbolID) + r" . Price ($\$$)"
    ylabel =   ul5.period_dic[myTimeData.period]
    
    
#    fig = gl.init_figure()
    ax = gl.tradingBarChart(myTimeData,  legend = ["Close price"], color = "k", 
                nf = 1, labels = [title,"",ylabel], AxesStyle = "Normal" + AxesStyle, 
                dataTransform = dataTransform)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)


redraw_chart(myTimeData)
plt.pause(10)
"""
############# Initial chart ##################
"""

###### Server options #################
mode = "Blocking"
port = 9096
serv = sockets_lib.socketserver('127.0.0.1',port)
print("TCP Server listening on port ", port, ", mode = ", mode)
serv.listen();

# Ask for the list of symbols! 
#success = serv.request_csv_symbol_info();
#
#if (success):
#    Symbol_info = CSy.load_symbols_info(storage_folder)
#    CSy.save_symbols_info(storage_folder, Symbol_info)
#    print(Symbol_info)
#    
#    ## Now we download data from the first 10 symbols and mix it with previous one.
#    periods = [1, 5, 15, 1440]
#    Symbol_names = Symbol_info["Symbol"].tolist()
#    Nsym = len(Symbol_names)
#    
#    ## Download the last week ofsome symbols
#    sdate = dt.date.today() - dt.timedelta(days=7000)
#    sdate = sdate.strftime("%d %m %Y")
#    
#    for i in range(Nsym):
#        for p in range(len(periods)):
#            symbolID = Symbol_names[i]
#            period = periods[p]
#            success = serv.request_csv_data_signal(symbolID, period,sdate);
#            
#            if (success):
#                ## Load the local and new data and save it !!
#                timeData = CTD.CTimeData(symbolID,period)
#                timeData.update_csv (storage_folder, updates_folder)
#                print ("Updated database")
#                  
#serv.sock.close()



#if(0):
#    # Listening to the data
#    while True:  
#        print ("Listening")
#        msg = serv.recvmsg()
