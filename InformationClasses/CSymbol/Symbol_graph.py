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
from datetime import datetime

######################################################################
############# BASIC PLOTS #######################################
######################################################################

def plot_timeSeries(self):
    # TODO make it possible to plot dates in the X axis
    labels = ["Price Value", "Time", self.value_names, self.value_names]
    gr.plot_graph([],
                  self.timeSeries.T,
                  labels, 1)

def plot_timeSeriesReturn(self):
    if (self.timeSeriesReturn == []):
        self.get_timeSeriesReturn();
        
    labels = ["Returns Value", "Time", self.value_names, self.value_names]
    gr.plot_graph([],
                  self.timeSeriesReturn.T,
                  labels, 1)

def plot_timeSeriesCumReturn(self):
    if (self.timeSeriesCumReturn == []):
        self.get_timeSeriesCumReturn();
        
    labels = ["Returns Value", "Time", self.value_names, self.value_names]
    gr.plot_graph([],
                  self.timeSeriesCumReturn.T,
                  labels, 1)

######################################################################
############# BASIC Veleros #######################################
######################################################################

def plot_dailyVJ(self, ini = 0, fin = -1):
    ## PLOTS DAILY Velas Japonesas

    data = np.matrix(self.TD[['Close', 'Open', 'High', 'Low']][ini:fin].values).T
    volume = self.TD['Volume'][ini:fin].values
#    print data
    labels = ["Velas Japonesas","Day","Price",self.symbol]
    gr.Velero_graph([], data, volume, labels,new_fig = 1)
    
def plot_dailyHA(self, ini = 0, fin = -1):
    ## PLOTS DAILY HEIKE ASHI

    data = self.TD[['Open', 'High', 'Low', 'Close']][ini:fin]
    volume = self.TD['Volume'][ini:fin].values

    labels = ["Heiken Ashi","Day","Price",self.symbol]
    gr.Heiken_Ashi_graph([], data, volume, labels,new_fig = 1)
    
######################################################################
############# Specific Graphs #######################################
######################################################################
    
def scatter_deltaDailyMagic(self):
    ## PLOTS DAILY HEIKE ASHI
    ddelta = self.get_dailyDelta()
    mdelta = self.get_magicDelta()
    labels = ["Delta Magic Scatter","Magic","Delta",self.symbol]
    gr.scatter_graph(mdelta,ddelta, labels,new_fig = 1)
    

######################################################################
############# Moving Averages Graph #######################################
######################################################################

def plot_TrCrMr(self):
    ## PLOTS DAILY HEIKE ASHI
    self.get_dailyPrice()
    TrCrMr = self.get_TrCrMr()
    labels = ["Delta Magic Scatter","Magic","Delta",self.symbol]
    gr.plot_graph([],self.dailyPrice, labels,new_fig = 1)
    
    labels = ["Delta Magic Scatter","Magic","Delta",self.symbol]
    gr.plot_graph([],TrCrMr, labels,new_fig = 0)

def plot_MA(self, flags):
    """ Function that plots the price and the indicated Moving Averages """
    
    self.get_dailyPrice()

    labels = ["Delta Magic Scatter","Magic","Delta",self.symbol]
    gr.plot_graph([],self.dailyPrice.T, labels,new_fig = 1)
    
    if (flags == 5):
        HMA = self.get_HMA(200)
        labels = ["Delta Magic Scatter","Magic","Delta",self.symbol]
        gr.plot_graph([],HMA, labels,new_fig = 0)
        
    if (flags == 3):
        ATR = self.get_ATR()
        print ATR.shape, self.dailyPrice.shape
        labels = ["Delta Magic Scatter","Magic","Delta"]
        gr.plot_graph([], np.matrix(self.dailyPrice).T + ATR.T, labels,new_fig = 0)
        
######################################################################
############# ELABORATE PLOTS #######################################
######################################################################
        
def plot_TD(self,start_date = [], end_date = []):
    
    labels = ['Close value', "Days", 'Adjusted Close']
    
    ### GET THE SUBSELECTION OF DAYS DATA ###
    if (start_date != []) and (end_date != []):  # If we specify a date
    
        start_date = np.datetime64(start_date).astype(datetime)
        end_date = np.datetime64(end_date).astype(datetime)
 
        index_time_list = self.TD.index.date  # Obtain the date only
#            print index_time_list
        
        indexs = np.where( index_time_list < end_date) and np.where( index_time_list > start_date)
        
#            print indexs
        
        # Subselection of the indexes 
        Close = self.intraData.ix[indexs]["Close"].values
        
    else:
        Close = self.intraData["Close"].values
        
    gr.plot_graph([],Close.T,labels, 1)
