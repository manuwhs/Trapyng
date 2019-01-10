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

import datetime as dt
from datetime import datetime

"""
Library with all the obtaining indicator functions of the market.

"""

# Start Date is the date from which we return the data.
# The data returned should be returned after this date.

# TimeSeries is the main data we have to care about. 
# All the operations will be done over this one

def set_date(self, date):
    self.imaginaryDate = date;
    self.Coliseum.set_date(date)
    
def process_new_actions (self, buy_symbols, sell_symbols):
    # This function sends the action to perform [BUY,SELL,HOLD] for a set of symbols

    for symbol in  buy_symbols:  # For every symbol told to BUY
    
        ### FIRST WE CLOSE ALL Open SELL positions for that symbol
        open_positions = self.Coliseum.get_positions_symbol(symbol)         # Get current open positions
        open_positions = open_positions[open_positions["Type"] == "SELL"]    # 
        self.Coliseum.close_positions(open_positions)
        
        ## TODO get the size of inversion in terms of the type of action
        self.Coliseum.open_position(symbol, "BUY", 20)
        
    for symbol in sell_symbols:  # For every symbol told to SELL
        ### FIRST WE CLOSE ALL Open BUY positions for that symbol
        open_positions = self.Coliseum.get_positions_symbol(symbol)         # Get current open positions
        open_positions = open_positions[open_positions["Type"] == "BUY"]    # 
        self.Coliseum.close_positions(open_positions)
        
        ## TODO get the size of inversion in terms of the type of action
        self.Coliseum.open_position(symbol, "SELL", 20)