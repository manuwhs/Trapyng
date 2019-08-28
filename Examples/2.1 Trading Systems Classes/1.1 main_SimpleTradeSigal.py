"""

In this document we show how a simple Entry trading strategy works,
in this case we set a Crossing Average technique with 2 EMAs and plot 
the events
 
"""

import os
os.chdir("../../")
import import_folders
# Classical Libraries

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
# Own graphical library
from graph_lib import gl 
# Data Structures Data


import CPortfolio as CPfl
import CStrategy_XingMA as CX
# Import functions independent of DataStructure
import utilities_MQL5 as ul5
import utilities_lib as ul


plt.close("all")
######## SELECT DATASET, SYMBOLS AND PERIODS ########
source = "MQL5" # Hanseatic  FxPro GCI Yahoo
[storage_folder, updates_folder] = ul5.get_foldersData(source = source)
################## Date info ###################
sdate_str = "01-11-2016"; edate_str = "21-12-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
symbolIDs =  ["EURUSD","USDCAD"]
periods = [1440]  # 1440  43200
period1 = periods[0]

####### LOAD SYMBOLS AND SET Properties   ###################
Cartera = CPfl.Portfolio("MyPF",symbolIDs, periods)   # Set the symbols and periods to load
# Download if needed.
#Cartera.update_symbols_csv_yahoo(sdate_str,edate_str,storage_folder)    # Load the symbols and periods
Cartera.set_csv(storage_folder)    # Load the symbols and periods
## SET THINGS FOR ALL OF THEM
Cartera.set_interval(sdate,edate)
Cartera.set_seriesNames(["Close"])

########## Strategy setting ####################
myEstrategia = CX.CStrategy_XingMA("caca",1440,Cartera)
symbolID = "EURUSD"
Lslow, Lfast = 30,5
myEstrategia.set_slowMA(symbolID, period1, L = Lslow, MAtype = "EMA")
myEstrategia.set_fastMA(symbolID, period1, L = Lfast, MAtype = "EMA")
crosses, dates = myEstrategia.get_TradeSignals()

########## Signals simulation ####################
EMAslow = Cartera.EMA(symbolIDs = [symbolID], period = period1, n = Lslow)[0]
EMAfast = Cartera.EMA(symbolIDs = [symbolID], period = period1, n = Lfast)[0]
price = Cartera.get_timeData(symbolID, period1).get_timeSeries()
dataHLOC = Cartera.get_timeData(symbolID, period1).get_timeSeries(["High","Low","Open","Close"])
dates = Cartera.get_timeData(symbolID, period1).get_dates()

########## Plotting ! ####################
gl.set_subplots(2,1)
ax1 = gl.plot(dates,price, legend = ["Price"],
        labels = ["Crossing MA Strategy", "","Price"], nf = 1)
gl.plot(dates,EMAslow, legend = ["Slow"], lw = 3)
gl.plot(dates,EMAfast, legend = ["fast"])


ax2 = gl.plot(dates, ul.scale(EMAfast  - EMAslow), legend = ["Difference"]
        ,nf = 1, sharex = ax1, labels = ["","","Events"], fill = 1, alpha = 0.3)
gl.stem(dates,crosses, legend = ["TradeSignal"], )
gl.scatter(dates, ul.scale(EMAfast  - EMAslow), lw = 0.5, alpha = 0.5)

gl.plot(dates,np.zeros(crosses.shape))
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.10, hspace=0.01)

trade_events = myEstrategia.get_TradeEvents()
 
# We can observe the apparent delay !

#crosses, dates = Hitler.RobustXingAverages(symbols[0], Ls_list,Ll_list)  # 

