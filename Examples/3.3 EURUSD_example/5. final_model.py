"""
This file loads a previosly trained model, loads the data file indicated,
preprocess it, computes the estimated probabilities and creates a .csv file
with the predictions in the correct original time
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
import t_utils as tut
import baseClassifiersLib as bCL
import basicMathlib as bMl
import indicators_lib as indl
import pickle_lib as pkl 
import utilities_lib as ul
plt.close("all") # Close all previous Windows

# %% 
"""
################### EXECUTING OPTIONS ###################
"""

folder_images = "../pics/EURUSD/"
storage_folder = ".././storage/EURUSD/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

load_data = 1
preprocessing_data = 1
extract_features = 1



# Using the library of function built in using the dataFrames in pandas
typeChart = "Bar"  # Line, Bar, CandleStick
tranformIntraday = 1

symbols = ["EURUSD"]
periods = [15]  # 1440 15


# %% 
if (load_data):
    ######## CREATE THE OBJECT AND LOAD THE DATA ##########
    # Tell which company and which period we want
    timeData = CTD.CTimeData(symbols[0],periods[0])
    timeData.set_csv(storage_folder,file_name)  # Load the data into the model

if (preprocessing_data):
    timeData = tut.preprocess_data(timeData)
    ## Get the valid trading days sorted
    days_keys, day_dict = timeData.get_indexDictByDay()
    Ndays = len(days_keys)

    timeData_daily = tut.get_daily_timedata(timeData, symbols[0])
    H,L,O,C,V = np.array(timeData_daily.TD[["High","Low","Open","Close","Volume"]][:]).T
    
if (extract_features):
    
    """
    Create the target for the regressing and classifier systems
    """
    Target = bMl.diff(C).flatten()  # Continuous target .Target[0] = NaN
    Target_bin = np.zeros(Target.shape) # Binarized target
    Target_bin[np.where(Target >=0)] = 1
    data_df = None
    
    ## Create Pandas Data Frame for the information of the ML problem
    data_df = pd.DataFrame({'Time': days_keys, 'Target_clas': Target_bin,  'Target_reg': Target})
    data_df.set_index('Time',inplace = True)
    
    ## Every feature that is computed for each day has to be lagged before introducing 
    ## it into the dataframe so that the input features do not contain information
    ## from their target day
    
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
    Nlag_trading_info = 1
    tut.add_lagged_values(data_df,HMA,"HMA",Nlag_trading_info)
    tut.add_lagged_values(data_df,RSI,"RSI",Nlag_trading_info)
    tut.add_lagged_values(data_df,MACD,"MACD",Nlag_trading_info)
    tut.add_lagged_values(data_df,ACCDIST,"ACCDIST",Nlag_trading_info)
    
    # %% 
    """
    Final Subselection of Features, normalization and data splitting
    """
    data_df = data_df[["Target_clas","Target_reg","week_1","Target_1","Target_2",
                       "Target_3","RSI_1","MACD_1","ACCDIST_1"]]
    # Remove the samples that did not have enough previous data !!!
    
    data_df["Original Time"] = timeData_daily.TD["Original Time"]
    data_df.dropna(inplace = True)
    dates_predictions = data_df["Original Time"]
    data_df.drop("Original Time",axis = 1, inplace =True)
    
    
    input_features_names = data_df.columns[np.logical_and(data_df.columns != 'Target_clas' , 
                                                          data_df.columns != 'Target_reg')]
    
    X = np.array(data_df.loc[:,input_features_names])
    Y = np.array(data_df.loc[:,'Target_clas']).reshape(-1,1)
    

## Normalize variables !!! 
from sklearn import preprocessing
"""
Load the model and the scalers
"""


folder_model = "../models/"
folder_predictions = "../predictions/"
key_classifier = "LSVM"  # QDA  # GNB RF

ul.create_folder_if_needed(folder_predictions)
scaler_X = pkl.load_pickle(folder_model + "scaler_X" +".pkl")[0]
scaler_Y = pkl.load_pickle(folder_model + "scaler_Y" +".pkl")[0]
classifier = pkl.load_pickle(folder_model + key_classifier +".pkl")[0]

X = scaler_X.transform(X)            
Y = scaler_Y.transform(Y)

Ypred = classifier.predict_proba(X)[:,1]
output_df = tut.get_output_df(dates_predictions,Ypred)

output_df = output_df[output_df.index > dt.datetime(2015,4,1)]
output_df.to_csv(folder_predictions + key_classifier + ".csv")

## Reload predictions for checking
loaded_predictions = pd.read_csv(folder_predictions + key_classifier + ".csv",  
                                 sep = ',', index_col = 0, header = 0)
loaded_predictions.index = pd.to_datetime(loaded_predictions.index)



    
