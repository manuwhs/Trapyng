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
import CStrategyPool as CSP
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
period2 = periods[0]
####### LOAD SYMBOLS AND SET Properties   ###################
Cartera = CPfl.Portfolio(symbols, periods)   # Set the symbols and periods to load
# Download if needed.
#Cartera.update_symbols_csv_yahoo(sdate_str,edate_str,storage_folder)    # Load the symbols and periods
Cartera.load_symbols_csv(storage_folder)    # Load the symbols and periods
## SET THINGS FOR ALL OF THEM
Cartera.set_interval(sdate,edate)
Cartera.set_seriesNames(["Close"])

#################### #################### #################### ####################

########## Strategy 1 ####################
Strategy1_ID = "DanskeStrategy"
myEstrategia = CX.CStrategy_XingMA(Strategy1_ID,1440,Cartera)
symbol1 = "DANSKE.CO"
Lslow, Lfast = 12,5
myEstrategia.set_slowMA(symbol, period1, L = Lslow, MAtype = "EMA")
myEstrategia.set_fastMA(symbol, period1, L = Lfast, MAtype = "EMA")

########## Strategy 2 ####################
Strategy2_ID = "GEStratefy"
myEstrategia2 = CX.CStrategy_XingMA(Strategy2_ID,1440,Cartera)
symbol2 = "GE"
Lslow, Lfast = 20,8
myEstrategia.set_slowMA(symbol2, period2, L = Lslow, MAtype = "EMA")
myEstrategia.set_fastMA(symbol2, period2, L = Lfast, MAtype = "EMA")

########### Strategy Pool #################
myPool = CSP.CStrategyPool()
myPool.add_strategy(Strategy1_ID, Strategy1_ID)
myPool.add_strategy(Strategy2_ID, Strategy2_ID)

EntrySignals = myEstrategia.get_TradeEvents()

#################### #################### #################### ####################
# Now that we are all set, lets plot both strategies and signals

########## Signals simulation ####################
EMAslow = Cartera.symbols[symbol1].TDs[period1].EMA(n = Lslow)
EMAfast = Cartera.symbols[symbol1].TDs[period1].EMA(n = Lfast)
price = Cartera.symbols[symbol1].TDs[period1].get_timeSeries()
dataHLOC = Cartera.symbols[symbol1].TDs[period1].get_timeSeries(["High","Low","Open","Close"])
dates = Cartera.symbols[symbol1].TDs[period1].get_dates()

########## Plotting ! ####################
gl.set_subplots(4,1)

## Plot both prices and EMAs
ax1 = gl.plot(dates,price, legend = ["Price"],
        labels = ["Crossing MA Strategy", "","Price"], alpha = 0, nf = 1)
        
title = "Bar Chart. " + str(symbols[0]) + "(" + ul.period_dic[timeData.period]+ ")" 
timeData = Cartera.symbols[]
gl.tradingBarChart(timeData, ax = ax1,  legend = ["Close price"], color = "k",
                    labels = [title,"",r"Price ($\$$)"], AxesStyle = "Normal - No xaxis")

gl.plot(dates,EMAslow, legend = ["Slow"], lw = 1, color = "b")
gl.plot(dates,EMAfast, legend = ["fast"], lw = 1, color = "r", xaxis_mode = "hidden")

## Entry Axes 
ax2 = gl.plot(dates, ul.scale(EMAfast  - EMAslow), legend = ["Difference"]
        ,nf = 1, sharex = ax1, labels = ["","","Events"], fill = 1, alpha = 0.3)
gl.stem(dates,crosses, legend = ["TradeSignal"] )
gl.scatter(dates, ul.scale(EMAfast  - EMAslow), lw = 0.5, alpha = 0.5)
gl.plot(dates,np.zeros(crosses.shape))


 
# We can observe the apparent delay !

#crosses, dates = Hitler.RobustXingAverages(symbols[0], Ls_list,Ll_list)  # 

