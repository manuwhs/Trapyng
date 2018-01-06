""" BASIC USAGE OF THE CLASS PORTFOLIO """
# Change main directory to the main folder and import folders
### THIS CLASS DOES NOT DO MUCH, A QUICK WAY TO AFFECT ALL THE TIMESERIES
## OF THE SYMBOL IN A CERTAIN MANNER, like loading, or interval.
## It also contains info about the Symbol Itself that might be relevant.
import os
os.chdir("../../")
import import_folders
# Classical Libraries
import copy as copy
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
# Own graphical library
from graph_lib import gl 
# Data Structures Data

import CSymbol as CSy
import CPortfolio as CPfl
import CStrategy_XingMA as CX
# Import functions independent of DataStructure
import utilities_lib as ul

plt.close("all")
######## SELECT DATASET, SYMBOLS AND PERIODS ########
source = "Yahoo" # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = source)
################## Date info ###################
sdate_str = "01-01-2010"
edate_str = "21-12-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")
symbolIDs =  ["GE", "HPQ","XOM","DANSKE.CO"]
periods = [43200]  # 1440  43200
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
symbolID = "DANSKE.CO"
Lslow, Lfast = 12,5
myEstrategia.set_slowMA(symbolID, period1, L = Lslow, MAtype = "EMA")
myEstrategia.set_fastMA(symbolID, period1, L = Lfast, MAtype = "EMA")
crosses, dates = myEstrategia.get_TradeSignals()

########## Signals simulation ####################
EMAslow = Cartera.get_timeData(symbolID,period1).EMA(n = Lslow)
EMAfast = Cartera.get_timeData(symbolID,period1).EMA(n = Lfast)
price = Cartera.get_timeData(symbolID,period1).get_timeSeries()
dates = Cartera.get_timeData(symbolID,period1).get_dates()

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

