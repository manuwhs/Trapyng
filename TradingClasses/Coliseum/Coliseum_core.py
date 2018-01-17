# -*- coding: utf-8 -*-
#import matplotlib

import numpy as np
import copy
import time
import pandas as pd
import graph_lib as gr
import Intraday_lib as itd
import utilities_lib as ul
import indicators_lib as indl
import get_data_lib as gdl 

import datetime as dt
from datetime import datetime

"""
Library with all the obtaining indicator functions of the market.

"""

""" $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Create Functions to load the real Coliseum from the MetaTrader and actualize it
We can calculate splippage and shit.

"""
# Start Date is the date from which we return the data.
# The data returned should be returned after this date.

# TimeSeries is the main data we have to care about. 
# All the operations will be done over this one


def load_csv(self, file_dir = "/"):
    # This function loads the Coliseum from a CSV
    whole_path = file_dir + "OrdersReport.csv"
    try:
        dataCSV = pd.read_csv(whole_path,
                          sep = ',')
    except IOError:
        error_msg = "File does not exist: " + whole_path 
        print (error_msg)
        dataCSV = ul.empty_coliseum
    
    return dataCSV


def set_date(self, date):
    self.imaginaryDate = date;
    
def get_positions_symbol(self,symbol):
    # Gets all the positions of a symbol
    Selected_pos = self.Warriors
    Selected_pos = Selected_pos[Selected_pos["Symbol"] == symbol]
    
    return Selected_pos
    
def get_position_indx(self,symbol,type_order, size):
    Selected_pos = self.Warriors
    Selected_pos = Selected_pos[Selected_pos["Symbol"] == symbol]
    Selected_pos = Selected_pos[Selected_pos["Size"] == size]
    Selected_pos = Selected_pos[Selected_pos["Type"] == type_order]

    indexes = Selected_pos.index.tolist()
    return indexes

def close_position_by_indx (self, indx):
    ## First we calculate the profit. For that we need to have the current price updated
    symbol_name = self.Warriors["Symbol"][indx]
    self.update_prices()

    ### Update Prices
#    self.freeMargin += self.Warriors["Symbol"][indx];   # Money we have not invested.
    self.moneyInvested = 0;  # Total of money invested in open positions
    self.marginLevel = 0;   # Nivel de apalanzamiento sobre el margen
    self.Equity = 0;       # How much money we would have if we close all positions

    self.Profit += self.Warriors["Profit"][indx]
    
    self.Warriors.drop(indx, inplace = True)
    if (len(self.Warriors.index.values ) > 0): # If there still warriors
        self.Warriors.reset_index( inplace = True)
    return indx;

def close_position (self, symbol,type_order, size):
    indx = self.get_position_indx(symbol,type_order, size)
    self.close_position_by_indx(indx)
    return indx;

def close_positions (self, positions):
    indexes = self.Warriors.index.tolist()
    indexes = sorted(indexes, reverse = True)
    for indx in indexes:
        self.close_position_by_indx(indx)
        
    return 1
    
def open_position (self, symbol,type_order, size ):

    # Calculat PriceOpen from self.symbol_info
    # ['Symbol','Type','Size','TimeOpen','PriceOpen', 'Comision','CurrentPrice','Profit'] 
    position_col = ul.empty_coliseum  # Create an empty coliseum
#    Npos_col = len(position_col.columns.tolis())
    
    
    position_col.loc[0] = [symbol, type_order, size, 
#                           dt.datetime.now(),
#                           self.Portfolio.symbols[symbol].get_currentPrice(),
                           self.imaginaryDate,
                           self.Portfolio.symbols[symbol].get_priceDatetime(self.imaginaryDate, 1440),
    
                         -1, -1, -1]
    self.add_position(position_col.ix[0])
    
def add_position(self,position):
    # Adds a position to the table
    #print 
    if (len(self.Warriors.index.values ) == 0): # If no warriors
        next_indx = 0;
    else:
        next_indx = self.Warriors.index.values[-1] + 1
    self.Warriors.loc[next_indx] = copy.deepcopy(position)

    self.update_prices()
    ## Update Prices !!!
    self.freeMargin -= position["Size"] * position["PriceOpen"];   # Money we have not invested.
    self.moneyInvested += position["Size"] * position["PriceOpen"];  # Total of money invested in open positions
    self.marginLevel = 0;   # Nivel de apalanzamiento sobre el margen

def update_prices(self):
    # This function updates the prices from the Portfolio
    indexes = self.Warriors.index.tolist()
    for indx in indexes:
        symbol_name = self.Warriors["Symbol"][indx]
#        current_price = self.Portfolio.symbols[symbol_name].get_currentPrice()
        current_price = self.Portfolio.symbols[symbol_name].get_priceDatetime(self.imaginaryDate, 1440)
        
        self.Warriors["CurrentPrice"][indx] = current_price
        ## TODO info about the price
        ## TODO add the commision
        if (self.Warriors["Type"][indx] == "BUY"):
            self.Warriors["Profit"][indx] = (self.Warriors["CurrentPrice"][indx] - self.Warriors["PriceOpen"][indx]) * self.Warriors["Size"][indx] 
        else:
            self.Warriors["Profit"][indx] = -(self.Warriors["CurrentPrice"][indx] - self.Warriors["PriceOpen"][indx]) * self.Warriors["Size"][indx] 
    
    return 1;

def get_commissions(self):
    """
    There are 3 types of commisions:
        1- Spread: D
        2- Operation: 0.05% for shares
        3- Swap: Cost for waiting from 1 day to the other 
    
    Given a colliseum:
        1- Spread is calculated when the position is opened
        2- Operation is calculated when the position is oppened
        3- Swap is calculated at 23:00 of each day
    
    """

def swap_comm(self):
    # Loops over the Colliseum and adds the commision
    # This function is to be called at 23:00 TC
    return 1

def spread_comm(self):
    # Whenever we open a position for backtesting we sustract or sum this value
    # to the current price. It's what the broker does. This commision does not 
    # have to be taken care of any more.

    # This function is to be called at 23:00 TC
    return 1

def operation_comm(self):
    # Whenever we open a position for backtesting we assign this cost.
    # It only applies for shares so we check the share.

    # This function is to be called at 23:00 TC
    return 1
    
def Update_from_Metatrader(self, prices):
    # This function updates the
    return 1;
    