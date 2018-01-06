# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 03:04:26 2016

@author: montoya
"""


import numpy as np
import datetime as dt
import CSymbol as CSy
import utilities_lib as ul

def default_select(self, symbolIDs, period):
    # Function that gives the default seleciton in case
    # we do not specify things
    if (type(symbolIDs) == type("str")):
        # We accept if only one str is given
        symbolIDs = [symbolIDs]
        
    elif (len(symbolIDs) == 0):
        symbolIDs = self.get_symbolIDs()
        
    if (type(period) == type(None)): # If no period specified we take all
#        print (self.symbols[symbolIDs[0]].get_periods())
        if (1440 in list(self.symbols[symbolIDs[0]].get_periods())):
            period = 1440
        else:
            period = max(self.symbols[symbolIDs[0]].get_periods())
    return symbolIDs, period

#################### Getting and deleting periods ###################
def init_symbols(self, symbolIDs = [], periods = [], symbols = []):
    # Function to initialice the Portfolio data.
    # If we are given a list of symbols, we just set them as the symbols
    # If we are not given that, but we are given a set of periods, we initialize
    # the Symbols with those periods, with empty data.
    if (len(symbolIDs) > 0):  # If there are symbols to initialize
        if (len(symbols) > 0): # If we are given the Symbol objects already
            self.add_symbols(symbols)
        else:
            if (len(periods) > 0): # If at least we are given some periods
                symbols = []                
                for symbolID in symbolIDs:   # Create the symbol objects
                    mySymbol = CSy.CSymbol(symbolID, periods)
                    symbols.append(mySymbol)
                self.add_symbols(symbols)
                
def get_symbolIDs(self):
    return self.symbols.keys()
    
#################### Getting and deleting symbols ###################
# Symbols will be a dictionary of [symbol]
# We will access them from the outside with these functions
def add_symbols(self, symbols = []):
    # Sets the secutities list
    for symbol_i in symbols:
        self.symbols[symbol_i.symbolID] = symbol_i
        
def set_symbols(self, symbols = []):
    # Sets the secutities list
    self.symbols = dict([]) # Remove all the previous symbols
    for symbol_i in symbols:
        self.symbols[symbol_i.symbolID] = symbol_i

def del_symbols(self, symbolIDs = []):
    # Sets the secutities list
    for symbol_i in symbolIDs:
        del self.symbols[symbol_i]
        
def get_symbols(self,symbolIDs = []):
    # returns a list of symbols objects given as names.
    # If we do not specify the symbols we want, we just get them all
    if (len(symbolIDs) == 0):
        symbolIDs = self.get_symbolIDs()
    symbols = []
    for symbol_n in symbolIDs:
        symbol = self.symbols[symbol_n]
        symbols.append(symbol)
    return symbols

#################### Getting and deleting timeDatas ###################
def get_timeData(self, symbolID, period):
    return self.symbols[symbolID].timeDatas[period]

############################################################
#############  Things to set to everyone ############# 
##############################################################
def set_seriesNames(self, seriesNames = []):
    for symbol_i in self.get_symbolIDs():
        self.symbols[symbol_i].set_seriesNames(seriesNames)

def set_interval(self,start_time = None, end_time = None):
    for symbol_n in self.get_symbolIDs():
        self.symbols[symbol_n].set_interval(start_time, end_time)

######################################################################
######################## Basic Interface to timeDatas ###############
######################################################################
def get_dates(self, symbolIDs = [], period = None):
    # This function gets the dates of one of the symbols
    # and one of the intervals
    # Returned as a list
    symbolIDs, period = self.default_select(symbolIDs, period)
    
    dates = []
    for symbol_n in symbolIDs:
        dates_n = self.symbols[symbol_n].timeDatas[period].get_dates()
        dates.append(dates_n)
    return dates
    
def get_timeSeries(self, symbolIDs = [], period = None, seriesNames = []):
    # This funciton returns a list with the timeSeries for all of the
    # symbols specified, for a given period.

    symbolIDs, period = self.default_select(symbolIDs, period)
    all_timeSeries = []

    for symbol_n in symbolIDs:
        timeSeries = self.symbols[symbol_n].timeDatas[period].get_timeSeries(seriesNames = seriesNames);
        all_timeSeries.append(timeSeries)
    return all_timeSeries

def get_timeSeriesReturn(self, symbolIDs = [], period = None, seriesNames = []):
    # This funciton returns a list with the timeSeries for all of the
    # symbols specified, for a given period.

    symbolIDs, period = self.default_select(symbolIDs, period)
    all_timeSeries = []
    for symbol_n in symbolIDs:
        timeSeries = self.symbols[symbol_n].timeDatas[period].get_timeSeriesReturn(seriesNames = seriesNames);
        all_timeSeries.append(timeSeries)
    return all_timeSeries
    
def get_timeSeriesCumReturn(self, symbolIDs = [], period = None, seriesNames = []):
    # This funciton returns a list with the timeSeries for all of the
    # symbols specified, for a given period.

    symbolIDs, period = self.default_select(symbolIDs, period)
    all_timeSeries = []
    for symbol_n in symbolIDs:
        timeSeries = self.symbols[symbol_n].timeDatas[period].get_timeSeriesCumReturn(seriesNames = seriesNames);
        all_timeSeries.append(timeSeries)
    return all_timeSeries

