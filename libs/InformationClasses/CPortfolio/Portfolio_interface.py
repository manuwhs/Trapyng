# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 03:04:26 2016

@author: montoya
"""


import numpy as np
import datetime as dt
import CSymbol as CSy
import graph_lib as gr

# Basically we want operations over all the symbols of a portfolio.
# for a given period type.


# secutities will be a dictionary of [symbol]

def get_intra_by_days(self, period):
    # Sets the secutities list
    
    days_list_price = [];
    days_list_date = [];
    for i in range(len(self.symbol_names)):
        symbol_i = self.symbol_names[i]
        symbol_period = self.symbols[symbol_i].TDs[period]
        
        day_intra_price, day_intra_dates = symbol_period.get_intra_by_days()
        days_list_price.append(day_intra_price)
        days_list_date.append(day_intra_dates)
    return days_list_price, days_list_date

          