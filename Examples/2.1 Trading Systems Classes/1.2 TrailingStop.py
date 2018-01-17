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

import CPortfolio as CPfl
import CStrategy_XingMA as CX
import CColiseum as CCO
import CTrailingStop as CTS
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
symbols =  ["GE", "HPQ","XOM","DANSKE.CO"]
periods = [43200]  # 1440  43200
period1 = periods[0]

####### LOAD SYMBOLS AND SET Properties   ###################
Cartera = CPfl.Portfolio("BEST_PF", symbols, periods)   # Set the symbols and periods to load
# Download if needed.
#Cartera.update_symbols_csv_yahoo(sdate_str,edate_str,storage_folder)    # Load the symbols and periods
Cartera.set_csv(storage_folder)    # Load the symbols and periods
## SET THINGS FOR ALL OF THEM
Cartera.set_interval(sdate,edate)
Cartera.set_seriesNames(["Close"])

########## Strategy setting ####################
myEstrategia = CX.CStrategy_XingMA("caca",1440,Cartera)
symbol = "DANSKE.CO"
Lslow, Lfast = 12,5
myEstrategia.set_slowMA(symbol, period1, L = Lslow, MAtype = "EMA")
myEstrategia.set_fastMA(symbol, period1, L = Lfast, MAtype = "EMA")
crosses, dates = myEstrategia.get_TradeSignals()
EntrySignals = myEstrategia.get_TradeEvents()

########## Signals simulation ####################
EMAslow = Cartera.EMA(symbolIDs = [symbol], period = period1, n = Lslow)[0]
EMAfast = Cartera.EMA(symbolIDs = [symbol], period = period1, n = Lfast)[0]
price = Cartera.get_timeData(symbol, period1).get_timeSeries()
dataHLOC = Cartera.get_timeData(symbol, period1).get_timeSeries(["High","Low","Open","Close"])
dates = Cartera.get_timeData(symbol, period1).get_dates()

####### Exit Policy setting ##########
symbol = "DANSKE.CO"
maxPullback = 25
# IDEA: Correlacion de volumenes en las rupturas.
# Correlacion volumenes de acciones de un indice
mySalida = CTS.CTrailingStop("salircaca",period1,Cartera)
firstEntrySingal = EntrySignals[0]
mySalida.set_trailingStop(SymbolName = symbol,BUYSELL = firstEntrySingal.BUYSELL, datetime = firstEntrySingal.datetime,
                          period = period1, maxPullback = maxPullback)
exitcrosses, all_stops, pCheckCross, datesExit = mySalida.get_TradeSignals()
# Second Signal
mySalida2 = CTS.CTrailingStop("salircaca",period1,Cartera)
secondEntrySingal = EntrySignals[1]
mySalida2.set_trailingStop(SymbolName = symbol,BUYSELL = secondEntrySingal.BUYSELL, datetime = secondEntrySingal.datetime,
                          period = period1, maxPullback = maxPullback)
exitcrosses2, all_stops2, pCheckCross2, datesExit2 = mySalida2.get_TradeSignals()

########## Plotting ! ####################
gl.set_subplots(3,1)
## Price Axes 
ax1 = gl.plot(dates,price, legend = ["Price"],
        labels = ["Crossing MA Strategy", "","Price"], alpha = 0, nf = 1)
        
gl.barchart(dates, dataHLOC)

gl.plot(dates,EMAslow, legend = ["Slow"], lw = 1, color = "b")
gl.plot(dates,EMAfast, legend = ["fast"], lw = 1, color = "r", xaxis_mode = "hidden")

## Entry Axes 
ax2 = gl.plot(dates, ul.scale(EMAfast  - EMAslow), legend = ["Difference"]
        ,nf = 1, sharex = ax1, labels = ["","","Events"], fill = 1, alpha = 0.3)
gl.stem(dates,crosses, legend = ["TradeSignal"] )
gl.scatter(dates, ul.scale(EMAfast  - EMAslow), lw = 0.5, alpha = 0.5)
gl.plot(dates,np.zeros(crosses.shape))

## Exit  Axes 
gl.add_hlines(datesExit, all_stops, ax = ax1, legend = ["TrailingStop"], alpha = 0.8, lw = 0.2)
ax3 = gl.plot(datesExit, ul.scale(-(all_stops - pCheckCross)), legend = ["TrailingStop"]
        ,nf = 1, sharex = ax1, labels = ["","","TrailingStop"],  lw = 0.5, alpha = 0.5, fill = 1)
gl.stem(datesExit,exitcrosses, legend = ["TradeSignal"] )
gl.scatter(datesExit, ul.scale(-(all_stops - pCheckCross)), lw = 0.5, alpha = 0.5)

gl.add_hlines(datesExit2, all_stops2, ax = ax1, legend = ["TrailingStop2"], alpha = 0.8, lw = 0.2)
gl.plot(datesExit2, ul.scale(all_stops2 - pCheckCross2), legend = ["TrailingStop2"],
        lw = 0.5, alpha = 0.5, fill = 1)
gl.stem(datesExit2,exitcrosses2, legend = ["TradeSignal"] )
gl.scatter(datesExit2, ul.scale(all_stops2 - pCheckCross2), lw = 0.5, alpha = 0.5)

gl.plot(dates,np.zeros(exitcrosses.shape))
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.10, hspace=0.01)

trade_events = myEstrategia.get_TradeEvents()
 
# We can observe the apparent delay !

#crosses, dates = Hitler.RobustXingAverages(symbols[0], Ls_list,Ll_list)  # 

