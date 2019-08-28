"""
This file is meant to update the storage folder by downloading all available data
from MT5 and combine it with the previously downloaded.
For this purpose it uses TCP sockets to connect to the corresponding MT5 client.

This file creates a socket Server.
When the correspoding MQL5 file is connected to it then:
    - Python asks MT5 for the list of symbols.
    - Python asks MT5 for the list of open positions.
    - Python asks MT5 for the OCHLV data of every symbol in several periods.
        - MT5 saves them in csv files
        - Python then reads them and combines them with the previously downloaded.
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

"""
####################### OPTIONS ###########################
"""

dataSource =  "MQL5"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, updates_folder] = ul5.get_foldersData(source = dataSource)
#### Load the info about the available symbols #####
Symbol_info = CSy.load_symbols_info(storage_folder)

## L
mode = "Blocking"
port = 9095
serv = sockets_lib.socketserver('127.0.0.1',port)
print("TCP Server listening on port ", port, ", mode = ", mode)
serv.listen();

# Ask for the list of symbols! 
success = serv.request_csv_symbol_info();

if (success):
    Symbol_info = CSy.load_symbols_info(updates_folder)
    CSy.save_symbols_info(storage_folder, Symbol_info)
    print(Symbol_info)
    
    ## Now we download data from the first 10 symbols and mix it with previous one.
    periods = [5] # [1, 5, 15, 1440]
    Symbol_names = Symbol_info["Symbol"].tolist()
    Nsym = len(Symbol_names)
    
    ## Download the last week ofsome symbols
    sdate = dt.date.today() - dt.timedelta(days=7000)
    sdate = sdate.strftime("%d %m %Y")
    
    for i in range(Nsym):
        print ("------ Downloading symbol %i/%i ----------"%(i+1,Nsym) )
        for p in range(len(periods)):
            symbolID = Symbol_names[i]
            period = periods[p]
            success = serv.request_csv_data_signal(symbolID, period,sdate);
            
            if (success):
                ## Load the local and new data and save it !!
                timeData = CTD.CTimeData(symbolID,period)
                timeData.update_csv (storage_folder, updates_folder)
                print ("Updated database")
                  
serv.sock.close()



#if(0):
#    # Listening to the data
#    while True:  
#        print ("Listening")
#        msg = serv.recvmsg()
