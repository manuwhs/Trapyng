"""
This file explores different features that can be extracted from the 
preprocessed data and creates a pandas dataframe with the windowed features.
"""
# %% 
# Load all the directories needed for the code to be executed both from
# a console and Spyder, and from the main directory and the local one.
# Ideally this code should be executed using Spyder with working directory

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Specific utilities
import indicators_lib as indl
import t_utils as tut
import basicMathlib as bMl

plt.close("all") # Close all previous Windows

# %%
"""
################### EXECUTING OPTIONS ###################
"""

folder_images = "../pics/EURUSD/"
storage_folder = ".././storage/EURUSD/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

# flags to load and preprocessed data (for rime optimization)
load_data = 1
preprocessing_data = 1

plotting_time_info = 0
plotting_OCHL_info = 0
plotting_Trading_indicators = 1
plotting_variables = 0

# Symbol information
symbols = ["EURUSD"]
periods = [15]  # 1440 15

######## SELECT DATE LIMITS ###########
## We set one or other as a function of the timeSpan
sdate_str = "01-01-2010"
edate_str = "31-12-2016"
#edate_str = "31-1-2011"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

# %%
if (load_data):
    ######## CREATE THE OBJECT AND LOAD THE DATA ##########
    # Tell which company and which period we want
    timeData = CTD.CTimeData(symbols[0],periods[0])
    timeData.set_csv(storage_folder,file_name)  # Load the data into the model

if (preprocessing_data):
    timeData = tut.preprocess_data(timeData, sdate,edate)
    ## Get the valid trading days sorted
    days_keys, day_dict = timeData.get_indexDictByDay()
    Ndays = len(days_keys)

    timeData_daily = tut.get_daily_timedata(timeData, symbols[0])
    H,L,O,C,V = np.array(timeData_daily.TD[["High","Low","Open","Close","Volume"]][:]).T
    
"""
Create the target for the regressing and classifier systems
"""
Target = bMl.diff(C).flatten()  # Continuous target .Target[0] = NaN
Target_bin = np.zeros(Target.shape) # Binarized target
Target_bin[np.where(Target >=0)] = 1

## Create Pandas Data Frame for the information of the ML problem
data_df = pd.DataFrame({'Time': days_keys, 'Target_clas': Target_bin,  'Target_reg': Target})
data_df.set_index('Time',inplace = True)

## Every feature that is computed for each day has to be lagged before introducing 
## it into the dataframe so that the input features do not contain information
## from their target day

# %%
################  Time specific variables ##################
# Used for algorithms to learn seasonal patterns and proximity between samples.
# We could just give the date but if we separate in different components it is 
# easier for algorithms to learn from it

day_of_week = np.array(data_df.index.dayofweek)
week_of_year = np.array(data_df.index.weekofyear)
year = np.array(data_df.index.year)

## Add the lagged value to the database
Nlag_time_information = 1
tut.add_lagged_values(data_df,day_of_week,"day",Nlag_time_information)
tut.add_lagged_values(data_df,week_of_year,"week",Nlag_time_information)
tut.add_lagged_values(data_df,year,"year",Nlag_time_information)


if (plotting_time_info):
    
    alpha_stem = 0.5
    marker_stem = [".",1,None]
    
    gl.init_figure();
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((4,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = gl.subplot2grid((4,1), (2,0), rowspan=1, colspan=1, sharex = ax1)
    ax4 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    
    ## Ax1 = Close price at the end of the sessions
    gl.plot(days_keys, C, ax = ax1, labels = ["Time variables","","Close Rate"],
            AxesStyle = "Normal - No xaxis", legend = ["Close Price"], color = "k")
    
    ## Ax2 = days
    gl.stem(days_keys, day_of_week, ax = ax2, labels = ["","","Day"], bottom = 0.0,
            AxesStyle = "Normal - No xaxis", alpha = alpha_stem,
            marker = marker_stem, color = "k", legend = ["Day"])

    # Ax3 = week
    gl.stem(days_keys,week_of_year ,bottom = 0.0, ax = ax3, labels = ["","","Week"],AxesStyle = "Normal - No xaxis", 
            alpha =alpha_stem, marker = marker_stem,  color = "k", legend = ["Week"])

    ## Ax4 = year
    gl.plot(days_keys,year, ax = ax4, labels = ["","","Year"],AxesStyle = "Normal", 
            alpha = alpha_stem,  color = "k", legend = ["Year"])
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    
    image_name = "time_info"
    gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 7])
    
# %%
################  OCHL variables ##################
# Variables trivially obtained from daily OHCL
Target = Target # Increase in Close price
Range_HL = H-L # measure of volatility
Daily_gap =  O - bMl.shift(C,lag = 1).flatten() # measure of price movement

## Add the lagged value to the database
Nlag_OCHL_information = 3
tut.add_lagged_values(data_df,Target,"Target",Nlag_OCHL_information)
tut.add_lagged_values(data_df,Range_HL,"Range_HL",Nlag_OCHL_information)
tut.add_lagged_values(data_df,Daily_gap,"Daily_gap",Nlag_OCHL_information)

if (plotting_OCHL_info):
    alpha_stem = 0.5
    marker_stem = [".",1,None]
    
    gl.init_figure();
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((4,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = gl.subplot2grid((4,1), (2,0), rowspan=1, colspan=1, sharex = ax1)
    ax4 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    
    ## Ax1 = Close price at the end of the sessions
    gl.plot(days_keys, C, ax = ax1, labels = ["OCHL variables","","Close Rate"],
            AxesStyle = "Normal - No xaxis", legend = ["Close Price"], color = "k")
    
    ## Ax2 = days
    gl.stem(days_keys, Target, ax = ax2, labels = ["","","Diff Close"], bottom = 0.0, 
            AxesStyle = "Normal - No xaxis", alpha = alpha_stem,
            marker = marker_stem, color = "k", legend = ["Diff Close"])

    # Ax3 = week
    gl.stem(days_keys,Range_HL ,bottom = 0.0, ax = ax3, labels = ["","","Range HL"],
            AxesStyle = "Normal - No xaxis", 
            alpha =alpha_stem, marker = marker_stem,  color = "k", legend = ["Range HL"],)

    ## Ax4 = year
    gl.stem(days_keys,Daily_gap, ax = ax4, labels = ["","","Daily gap"],AxesStyle = "Normal", 
            alpha = alpha_stem,  color = "k", legend = ["Daily gap"], marker = marker_stem,)
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    
    image_name = "OHCL_info"
    gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 7])
    
# %%
################## Daily Trading Indicators ####################
# Hulls_average !! ACDC, Volatility, ATR, Short 
nHMA = 20
## Hulls Average, reactive but smoothing MA
HMA  = indl.get_HMA(timeData_daily.get_timeSeries(["Close"]), nHMA)  

## Volatility
nAHLR = 20; nBB = 20; nATR = 20; nCha = 20;
AHLR = timeData_daily.AHLR(n = nAHLR)
ATR = timeData_daily.ATR(n = nATR)
EMA_Range, Cha = timeData_daily.Chaikin_vol(n = nCha)
BB = timeData_daily.BBANDS(seriesNames = ["Close"], n = nBB)
BB = BB[:,0] - BB[:,1] 

# Oscillators
n , SK, SD = 14, 6,6
L = 14
L1 , L2, L3 = 14, 9,12

STO = timeData_daily.STO(n = n, SK = SK, SD = SD)
RS, RSI = timeData_daily.RSI(n = L)
EX1,EX2,EX3,TRIX = timeData_daily.TRIX(L1 , L2, L3)
MACD, MACDsign, MACDdiff = timeData_daily.MACD().T
    
# Volume related
nAD = 5;
ACCDIST = timeData_daily.ACCDIST(n = nAD)
DV = timeData_daily.ACCDIST(n = nAD)

## Add variables to the data_frame
Nlag_trading_info = 3
tut.add_lagged_values(data_df,HMA,"HMA",Nlag_trading_info)
tut.add_lagged_values(data_df,RSI,"RSI",Nlag_trading_info)
tut.add_lagged_values(data_df,MACD,"MACD",Nlag_trading_info)
tut.add_lagged_values(data_df,ACCDIST,"ACCDIST",Nlag_trading_info)

if (plotting_Trading_indicators):
    ############# Volatility ############################
    alpha_stem = 0.7
    marker_stem = [".",1,None]
    image_name = "Volatility_info"
    
    gl.init_figure();
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((4,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = gl.subplot2grid((4,1), (2,0), rowspan=1, colspan=1, sharex = ax1)
    ax4 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    
    ## Ax1 = Close price at the end of the sessions
    gl.plot(days_keys, C, ax = ax1, labels = ["Volatility variables","","Close Rate"],
            AxesStyle = "Normal - No xaxis", legend = ["Close Price"], color = "k")
    gl.plot(days_keys, HMA, ax = ax1, labels = ["Volatility variables","","Hull MA"],
            AxesStyle = "Normal - No xaxis", legend = ["Hull MA(%i)"%nHMA], color = "b")
    ## Ax2
    gl.plot(days_keys, ATR, ax = ax2, labels = ["","","ATR"], 
            AxesStyle = "Normal - No xaxis", alpha = alpha_stem,
             color = None, legend = ["ATR(%i)"%nATR], fill = 0)
    # Ax3 
    gl.plot(days_keys,Cha ,ax = ax3, labels = ["","","Chaikin"],AxesStyle = "Normal - No xaxis", 
            alpha =alpha_stem, color = "k", legend = ["Chaikin(%i)"%(nCha)],fill = 1)
    ## Ax4 
    gl.plot(days_keys,BB, ax = ax4, labels = ["","","BB"],AxesStyle = "Normal", 
            alpha = alpha_stem,  color = "k", legend = ["BB(%i)"%nBB], fill = 0)
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 7])
    
    ############# Osillators ############################
    alpha_stem = 0.7
    marker_stem = [".",1,None]
    image_name = "Oscillators_info"
    
    gl.init_figure();
    ax1 = gl.subplot2grid((5,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((5,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = gl.subplot2grid((5,1), (2,0), rowspan=1, colspan=1, sharex = ax1)
    ax4 = gl.subplot2grid((5,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    ax5 = gl.subplot2grid((5,1), (4,0), rowspan=1, colspan=1, sharex = ax1)
    
    ## Ax1 = Close price at the end of the sessions
    gl.plot(days_keys, C, ax = ax1, labels = ["Oscillators variables","","Close Rate"],
            AxesStyle = "Normal - No xaxis", legend = ["Close Price"], color = "k")
    gl.plot(days_keys, HMA, ax = ax1, labels = ["","","Hull MA"],
            AxesStyle = "Normal - No xaxis", legend = ["Hull MA(%i)"%nHMA], color = "b")
    ## Ax2
    gl.plot(days_keys, STO, ax = ax2, labels = ["","","STO"], 
            AxesStyle = "Normal - No xaxis", alpha = alpha_stem,
             color = "k", legend = ["STOk(%i,%i)"%(n,SK), "STOd(%i)"%(SD)])
    # Ax3 
    gl.plot(days_keys,RSI ,ax = ax3, labels = ["","","RSI"],AxesStyle = "Normal - No xaxis", 
            alpha =alpha_stem, color = "k", legend = ["RSI"], fill = 1, fill_offset = 50)
    ## Ax4 
    gl.plot(days_keys,TRIX, ax = ax4, labels = ["","","TRIX"],AxesStyle = "Normal", 
            alpha = alpha_stem,  color = "k", legend = ["TRIX"])
    ## Ax4 
    gl.plot(days_keys,-MACD, ax = ax5, labels = ["","","MACD"],AxesStyle = "Normal", 
            alpha = alpha_stem,  color = "k", legend = ["MACD"])
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4,ax5], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 9])

    ############# Volume ############################
    alpha_stem = 0.7
    marker_stem = [".",1,None]
    image_name = "Volume_info"
    
    gl.init_figure();
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((4,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = gl.subplot2grid((4,1), (2,0), rowspan=1, colspan=1, sharex = ax1)
    ax4 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    
    ## Ax1 = Close price at the end of the sessions
    gl.plot(days_keys, C, ax = ax1, labels = ["Volume variables","","Close Rate"],
            AxesStyle = "Normal - No xaxis", legend = ["Close Price"], color = "k")

    ## Ax2
    gl.stem(days_keys, V, ax = ax2, labels = ["","","Volume"], 
            AxesStyle = "Normal - No xaxis", alpha = alpha_stem,
             color = None, legend = ["Volume"])
    # Ax3 
    gl.plot(days_keys,ACCDIST ,ax = ax3, labels = ["","","ACCDIST"],AxesStyle = "Normal - No xaxis", 
            alpha =alpha_stem, color = "k", legend = ["ACCDIST(%i)"%(nAD)])
    ## Ax4 
    gl.plot(days_keys,ACCDIST, ax = ax4, labels = ["","","ACCDIST"],AxesStyle = "Normal", 
            alpha = alpha_stem,  color = "k", legend = ["ACCDIST"], fill = 0)
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 7])

# %%
if(plotting_variables):
    from pandas.plotting import scatter_matrix
    data_df["Target_reg"] = (data_df["Target_reg"] - np.mean(data_df["Target_reg"]))/np.std(data_df["Target_reg"])
    scatter_matrix(data_df[["Target_reg","day_1","week_1","Target_1","Daily_gap_1","HMA_1"]])
    # scatter_matrix(data_df_train[["Target_reg","day_1","week_1","Range_HL_1","Target_1",
#                                    "Daily_gap_1","HMA_1","RSI_1","MACD_1","ACCDIST_1"]])
    plt.show()
    plt.gcf().set_size_inches( 10, 10 )
    plt.savefig(folder_images +'variables_1.png', dpi = 100) ## Variables

    scatter_matrix(data_df[["Target_reg","week_1","Target_1","Target_2","Target_3","RSI_1","MACD_1","ACCDIST_1"]])
    # scatter_matrix(data_df_train[["Target_reg","day_1","week_1","Range_HL_1","Target_1",
#                                    "Daily_gap_1","HMA_1","RSI_1","MACD_1","ACCDIST_1"]])
    plt.show()
    plt.gcf().set_size_inches( 10, 10 )
    plt.savefig(folder_images +'variables_2.png', dpi = 100) ## Variables
    
    

        
