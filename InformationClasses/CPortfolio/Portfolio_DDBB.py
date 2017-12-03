# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 03:04:26 2016

@author: montoya
"""

import pandas as pd
import numpy as np
import urllib2
import datetime as dt
import matplotlib.pyplot as plt
import copy as copy
import time as time

from pandas_datareader import wb
import datetime
import gc
import CSymbol as CSy

def load_symbols_info(self, file_dir = "./storage/"):
    # Load the info of all the symbols in the portfoolio

   Symbol_info = CSy.load_symbols_info(file_dir)
   for sym_i in self.symbol_names:   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.set_info(Symbol_info)
       
   return Symbol_info

## Operations from a list of symbols 
def update_symbols_csv(self, file_dir_current = "./storage/", file_dir_new = "../Trader/MQL4/Files/"):
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.update_TDs(file_dir_current, file_dir_new)
       gc.collect()  # Remove the unreachable space
       
def set_csv(self, file_dir = "./storage/"):
    # Load symbols from disk
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.set_csv(file_dir)

def add_csv(self, file_dir = "./storage/"):
    # Load symbols from disk
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.add_csv(file_dir)
       
def save_to_csv(self, file_dir = "./storage/"):
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.save_to_csv(file_dir)
       
########## DOWNLOAD DATA FROM GOOGLE ############################
def set_symbols_from_google(self, timeInterval = "30d"):
    # Loads a CSV and adds its values to the main structure
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.set_TDs_from_google(timeInterval = timeInterval)
       
### Download the symbols from Yahoo and update the ones we already have
def download_symbols_csv_yahoo(self, sdate,edate, file_dir_current = "./storage/"):
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.download_TDs_yahoo(sdate,edate,file_dir_current)
       gc.collect()  # Re

def update_symbols_csv_yahoo(self, sdate,edate, file_dir_current = "./storage/"):
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.update_TDs_yahoo(sdate,edate,file_dir_current)
       gc.collect()  # Remove the unreachable space
           
def fill_data(self):
   for sym_i in self.get_symbolIDs():   # Iterate over the securities
       symbol = self.symbols[sym_i]
       symbol.fill_data()
    