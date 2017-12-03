# -*- coding: utf-8 -*-
#import matplotlib

import numpy as np

import time
import pandas as pd
import graph_lib as gr
import Intraday_lib as itd
import utilities_lib as ul
import indicators_lib as indl
import get_data_lib as gdl 
import copy
import datetime as dt
from datetime import datetime
import CTimeData as CTD
"""
Library with all the obtaining indicator functions of the market.

"""

# Start Date is the date from which we return the data.
# The data returned should be returned after this date.

# TimeSeries is the main data we have to care about. 
# All the operations will be done over this one


########### Initialization functions ##############
def init_timeDatas(self,symbolID = None, periods = []):
    symbolID,periods = self.get_final_SymbolID_periods(symbolID,periods)
    # Initialize the timeDataObjects if we have the list of period that
    # we want to have
    for period in periods:  # Creates emppty Dataframes
        timeData = CTD.CTimeData(symbolID, period, ul.empty_df);
        self.add_timeData(period, timeData)
        
def get_final_SymbolID_periods(self, symbolID = None, periods = []):
    # Function to be used to check if we have been given enough information
    # It also sets the final values of the object as the resulting ones
    if (type(symbolID) == type(None)):
        if (type(self.symbolID) == type(None)):
            raise ValueError('No symbolID specified')
        else:
            symbolID = self.symbolID
    
    if (len(periods) == 0):
        if (len(self.get_periods()) == 0):
            raise ValueError('No periods specified')
        else:
            periods = self.get_periods()
    return symbolID, periods
    
###############################################################
######## Functions regarding info of the symbol #################
##############################################################

def get_periods (self):
    return self.timeDatas.keys()
######################################################################
######################## Interface functions to timeDatas ###############
######################################################################
def get_timeData (self, period = 1440):
    return self.timeDatas[period]

def add_timeData(self, period, timeDataObj):
    # This function adds the timeData object to the Symbol.
    # The timeDataObj is already an intialized timeDataObj.
    # We actually do not need to specify the period as we could
    # get it from the timeDataObj but this is more visual
    self.timeDatas[period] = timeDataObj

def del_timeData(self, period):
    # This function deletes the timeData object to the Symbol.
    del self.timeDatas[period] 
    
#    CTD.CTimeData(self.symbol, period,timeData);

####################################################################
######## Functions to apply to all timeDatas #################
######################################################################

def set_interval(self,start_time = [], end_time = [], trim = True):
    for period in self.get_periods():  # For each available period
        self.timeDatas[period].set_interval(start_time, end_time, trim = trim)
        
def set_seriesNames(self, seriesNames = []):
    for period in self.get_periods():
        self.timeDatas[period].set_seriesNames(seriesNames)

######################################################################
######################## Basic Interface to timeDatas ###############
######################################################################

def get_currentPrice (self):
    # This function gets the current price from the lowest source
    minimumTimeScale = np.min(self.get_periods())
    currentPrice = self.timeDatas[minimumTimeScale].TD["Close"][-1]
    return currentPrice

def get_priceDatetime (self, datetime_ask, period):
    # This function gets the price for the given date in the given timeScale
    minimumTimeScale = np.min(self.get_periods())
    dates = self.timeDatas[period].TD.index
#    print datetime_ask
    good_dates = dates[dates == datetime_ask]
    good_prices = self.timeDatas[period].TD[dates == datetime_ask]["Close"]
    
#    print good_prices
    return good_prices[-1]
    
#########################################################
################# DATAFILLING ###########################
#########################################################

## This is the data filling function !!
## Define cases of datafilling !!