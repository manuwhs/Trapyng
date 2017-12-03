
"""
ValueError: DateFormatter found a value of x=0, which is an illegal date.  This usually occurs because you have not informed the axis that it is plotting dates, e.g., with ax.xaxis_date()
"""
""" BASIC USAGE OF THE CLASS"""
# Change main directory to the main folder and import folders
### THIS CLASS DOES NOT DO MUCH, A QUICK WAY TO AFFECT ALL THE TIMESERIES
## OF THE SYMBOL IN A CERTAIN MANNER, like loading, or interval.
## It also contains info about the Symbol Itself that might be relevant.
import os
os.chdir("../")
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
#### Load the info about the available symbols #####
#Symbol_info = CSy.load_symbols_info(info_folder)
# TODO: I think I have to do properly the loading of the info into the mdoel
####### SELECT SYMBOLS AND PERIODS #################
#symbols = ["Mad.ITX", "XAUUSD", "XAGUSD", "USA.IBM"]
#symbols = ["XAUUSD", "XAGUSD"]
#periods = [5,15,60,1440]
symbols =  ["GE", "HPQ","XOM","DANSKE.CO"]
periods = [43200]  # 1440  43200
period1 = periods[0]

####### LOAD SYMBOLS AND SET Properties   ###################
Cartera = CPfl.Portfolio(symbols, periods)   # Set the symbols and periods to load
# Download if needed.
#Cartera.update_symbols_csv_yahoo(sdate_str,edate_str,storage_folder)    # Load the symbols and periods
Cartera.load_symbols_csv(storage_folder)    # Load the symbols and periods
## SET THINGS FOR ALL OF THEM
Cartera.set_interval(sdate,edate)
Cartera.set_seriesNames(["Close"])
######################## FILLING DATA  ##########################
#print "Filling Data"
#Cartera.fill_data()
#print "Data Filled"
###### WE CAN OPERATE DIRECTLY ON SYMBOLS #################
mySymbol = Cartera.symbols["GE"]

basic_joint_info = 1 # Obtaining info for the same period
indicators_lib = 1  # Obtaining the indicators of the timeSeries

###### FUNCTIONS TO OPERATE ON THE SAME PERIOD #############
basic_joint_info = 1
if (basic_joint_info == 1):
    
    prices = Cartera.get_timeSeries(period1)
    returns = Cartera.get_Returns(period1)
    cumReturns = Cartera.get_CumReturns(period1)
    dates = Cartera.get_dates(period1, Cartera.symbol_names[0])
    
    gl.set_subplots(2,2)
    
    
    gl.plot(dates, prices[:,0], 
            labels = ["Prices Portfolio Assets","","Price"],
            legend = [Cartera.symbol_names[0]])
    gl.plot(dates, prices[:,1], 
            legend = [Cartera.symbol_names[1]], nf = 0, na = 1)
            
    gl.plot(dates, returns, 
            labels = ["Returns Portfolio Assets","","Return"],
            legend = Cartera.symbol_names)
    
    gl.plot(dates, cumReturns, 
            labels = ["cumReturns Portfolio Assets","","Return"],
            legend = Cartera.symbol_names)
            
    gl.scatter(returns[:,0], returns[:,1],
               labels = ["Correlation", Cartera.symbol_names[0], Cartera.symbol_names[1]])


if (indicators_lib == 1):
    SMA = Cartera.SMA(n = 20)
    #Exponential Moving Average  
    EMA = Cartera.EMA(n = 13)
    #Pivot Points, Supports and Resistances  
    PPSR = Cartera.PPSR()
    #Bollinger Bands  
    BBANDS = Cartera.BBANDS(n = 12)
    #Average True Range  
    ATR = Cartera.ATR()
    #Momentum  
    MOM = Cartera.MOM()
    #Rate of Change  
    ROC = Cartera.ROC()
    #Stochastic oscillator %D  
    STO = Cartera.STO()
    #Relative Strength Index  
    RSI = Cartera.RSI()
    #Average Directional Movement Index  
    ADX = Cartera.ADX()
    #Accumulation/Distribution  
    ACCDIST = Cartera.ACCDIST()
    #MACD, MACD Signal and MACD difference  
    MACD = Cartera.MACD()
    ## Oscillator similar to MACD
    TRIX = Cartera.TRIX()

    gl.plot([],SMA)


#######################################################

#timeDataObjs = Cartera.get_timeDataObj(period1)
#
#gl.init_figure()
#ax1 = gl.subplot2grid((5,4), (0,0), rowspan=3, colspan=4)
##ax1.xaxis_date()
#flag = 1
#for tdo in timeDataObjs:
#    gl.tradingPV(tdo)
#    if (flag):
#        ax = gl.twin_axes()
##        ax.xaxis_date()
#        flag = 0

#    ws = 50
#    gl.add_slider(args = {"wsize":ws}, plots_affected = [])
#    gl.add_hidebox()


if (0):
    
    def get_CCorr(symbol_list, period,date_start,date_end, pf, W):
        # Given a list of symbols and period, it calculated the cross-correlation matrix between them
        # W is the size of the window
        
        time_series_np = ul.get_cumReturn(time_series_np)
    #    time_series_np = ul.get_return(time_series_np)
        L,m = time_series_np.shape
        
        CC_w = np.zeros((m,m))  # Matrix of CCw for a given window (not used)
        All_CC_w = [];  # List of all possible CC_w
        
        for i in range (W):   # Include first W zeros because it is delayed
            All_CC_w.append(copy.deepcopy(CC_w)) 
            
            
        # Obtain the cross correlation matrix of the first window
        CC_w = np.corrcoef(time_series_np[:W,:].T)   
        All_CC_w.append(copy.deepcopy(CC_w))
    
        good_mode = 0;  # Do it quick and intelligent  
        
        if (good_mode):
            for i in range (1,L-W): # We calculate the rest by addind the new and resting the last
                adding = np.outer(time_series_np[i+W,:],time_series_np[i+W,:].T)
                substing = np.outer(time_series_np[i,:],(time_series_np[i,:]))
                print adding
                
                CC_w = CC_w + adding - substing
                All_CC_w.append(copy.deepcopy(CC_w))
        else:
            for i in range (1,L-W):
                CC_w = np.corrcoef(time_series_np[i:i+W,:].T)   
                All_CC_w.append(copy.deepcopy(CC_w))
        return All_CC_w
    
    period = 60;
    date_start = dt.datetime(2016,4,15)
    date_end = dt.datetime(2016,4,26)
    pf = Cartera
    symbol_list = ["XAUUSD", "XAGUSD"]
    
    W = 10
    CC_all = get_CCorr(symbol_list, period,date_start,date_end, pf, W)
    
    CC_tot = []
    for i in range(len(CC_all)):
        CC_tot.append(CC_all[i][0][1])
    
    
    TD1 = Cartera.symbols["XAUUSD"].TDs[period]
    TD2 = Cartera.symbols["XAGUSD"].TDs[period]
    
    price1 = TD1.get_timeSeriesCumReturn()*10
    price2 = TD2.get_timeSeriesCumReturn()*10