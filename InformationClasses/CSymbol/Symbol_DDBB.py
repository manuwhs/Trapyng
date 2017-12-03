# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from numpy import loadtxt

import time
import pandas as pd
import graph_lib as gr
import Intraday_lib as itd
import utilities_lib as ul
import indicators_lib as indl
import get_data_lib as gdl 

import datetime as dt
import CTimeData as CTD

#############################################################
################# BASIC FUNCTIONS FOR A GIVEN PERIOD ########
############################################################

def load_csv_timeData_period(self, file_dir = "./storage/", period = 1440):
    # This function loads from the csv file, the TD related to the symbol,
    # and the period specified.

    # The file must have the name:
    #  symbolName_TimeScale.csv
    whole_path = file_dir + self.symbol + "_" + ul.period_dic[period] + ".csv"
    try:
        dataCSV = pd.read_csv(file_dir + self.symbol + "_" + ul.period_dic[period] + ".csv",
                              sep = ',', index_col = 0, dtype = {"Date":dt.datetime})
        
        dataCSV.index = dataCSV.index.astype(dt.datetime)   # We transform the index to the real ones
        
    except IOError:
        error_msg = "File does not exist: " + whole_path 
        print error_msg
        dataCSV = ul.empty_df
        
    return dataCSV

#############################################################
################# REGARDING SYMBOL INFO  ########
############################################################

def load_info(self, file_dir = "./storage/"):
    # This functions loads the symbol info file, and gets the
    # information about this symbol and puts it into the structure
    whole_path = file_dir + "Symbol_info.csv"
    try:
        infoCSV = pd.read_csv(whole_path,
                              sep = ',')
    except IOError:
        error_msg = "Empty file: " + whole_path 
        print error_msg
    
    self.set_info(infoCSV)
    
    return infoCSV

def set_info(self, infoCSV):
    # Given the dataFrame with the info of symbols,
    # This function finds the symbol name and initializes the attibutes.
    # Symbol_name,PointSize,MinTickValue,ContractSize,Currency,PriceNow
    self.info = infoCSV.loc[infoCSV['Symbol_name'] == self.symbolID]
    return infoCSV
#def save_symbol_info(file_dir = "./storage/"):
#    # This functions loads the symbol info
#    
#        
#    return infoCSV
    
#############################################################
################# BASIC FUNCTIONS FOR ALL PERIODS  ########
############################################################

def set_csv(self,file_dir = "./storage/", symbolID = None, periods = []):
    # Loads a CSV and adds its values to the main structure
    symbolID, periods = self.get_final_SymbolID_periods(symbolID, periods)
    for period in self.get_periods():
        self.timeDatas[period].set_csv(file_dir, symbolID, period)
    
def add_csv(self,file_dir = "./storage/", symbolID = None, periods = []):
    # Loads a CSV and adds its values to the main structure
    for period in self.get_periods():
        self.timeDatas[period].add_csv(file_dir, symbolID, period)

def save_to_csv(self,file_dir = "./storage/"):
    # Loads a CSV and adds its values to the main structure
    for period in self.get_periods():
#        print period
        self.timeDatas[period].save_to_csv(file_dir)

###################################################3333#########################

def fill_data(self):
    for period in self.get_periods():
#        print period
        self.timeDatas[period].fill_data()
        

def update_csv (self,file_dir_current = "./storage/", file_dir_new = "../Trader/MQL4/Files/",
                symbolID = None, periods = []):
    self.set_csv(file_dir_current, symbolID, periods)
    self.add_csv(file_dir_new, symbolID, periods)
    self.save_to_csv(file_dir_current)

#######################################################################
############## Yahoo func ##########################################
#######################################################################

## Loads all the TDS of the symbol from yahoo
def download_TDs_yahoo(self,sdate,edate,file_dir = "./storage/"):
    # Loads a CSV and adds its values to the main structure
    for period in self.get_periods():
#        print period
        if (period == 1440):
            precision = "d"
        elif(period == 43200):
            precision = "m"
            
        self.timeDatas[period].download_from_yahoo(sdate,edate, precision = precision)

def set_TDs_from_google(self,symbolID = None, periods = [], timeInterval = "30d"):
    # Loads a CSV and adds its values to the main structure
    symbolID, periods = self.get_final_SymbolID_periods(symbolID, periods)
    for period in periods:
        self.timeDatas[period].set_TD_from_google(symbolID,period,timeInterval)

## Loads all the TDS of the symbol from yahoo and updates them
def update_TDs_yahoo(self,sdate,edate,file_dir = "./storage/"):
    # Loads a CSV and adds its values to the main structure
     self.download_TDs_yahoo(sdate,edate)
     self.add_csv(file_dir)  ## Add it to the one we already have
     self.save_to_csv(file_dir)

