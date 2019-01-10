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

from graph_lib import gl
######################################################################
############# BASIC PLOTS #######################################
######################################################################
    
def plot_timeSeries(self, nf = 1, na = 0):
    dates = self.dates
    timeSeries = self.get_timeSeries()

    gl.plot(dates, timeSeries, nf = nf,
            labels = [self.symbol + "(" + str(self.period) + ")", 
            "Time (" + str(self.period) + ")", "Prices"],
            legend = self.seriesNames, na = na)

def plot_timeSeriesReturn(self, nf = 1):
    dates = self.dates
    timeSeries = self.get_timeSeriesReturn()
    
    gl.plot(dates, timeSeries, nf = nf,
            labels = [self.symbol + "(" + str(self.period) + ")", 
            "Time (" + str(self.period) + ")", "Return Prices"],
            legend = self.seriesNames, fill = 1)

def plot_timeSeriesCumReturn(self, nf = 1):
    dates = self.dates
    timeSeries = self.get_timeSeriesCumReturn()
    
    gl.plot(dates, timeSeries, nf = nf,
            labels = [self.symbol + "(" + str(self.period) + ")", 
            "Time (" + str(self.period) + ")", "CumReturn Prices"],
            legend = self.seriesNames)
 
######################################################################
############# Specific Graphs #######################################
######################################################################
    
def scatter_deltaDailyMagic(self):
    ## PLOTS DAILY HEIKE ASHI
    ddelta = self.get_timeSeriesbyName("RangeCO")
    hldelta = self.get_timeSeriesbyName("RangeHL")
    
    mdelta = self.get_magicDelta()
    labels = ["Delta Magic Scatter","Magic","Delta"]

    gl.scatter(mdelta,ddelta, 
               labels = labels,
               legend = [self.symbolID],
               nf = 1)
    
#    gl.set_subplots(1,1)
    gl.scatter_3D(mdelta,ddelta, hldelta,
                   labels = labels,
                   legend = [self.symbolID],
                   nf = 1)

######################################################################
############# Moving Averages Graph #######################################
######################################################################

def plot_TrCrMr(self):
    ## Plots the Three deadly cross thingy.
    timeSeries = self.get_timeSeries()
    TrCrMr = self.get_TrCrMr()
    labels = ["Trio de la Muerte","Time","Price"]
    gl.plot(self.dates,timeSeries, 
                  labels = labels,
                  legend = ["Price"],
                  color = "k",nf = 1)
    
    gl.plot(self.dates,TrCrMr, 
                  labels = labels,
                  legend = ["Trio de la Muerte"],
                  color = "b",nf = 0)

def plot_BollingerBands(self, new_figure = 0, L = 21):
    if (new_figure == 1):
        self.new_plot(title = "Bollinger Bands", xlabel = "time", ylabel = "Close")
    
    SMA = self.get_SMA(L = L)
    BB = self.get_BollingerBand(L = L)
    
    self.plot_timeSeries()
    gl.plot(self.dates,SMA + BB, legend = ["SMA + BB"], nf = 0)
    gl.plot(self.dates,SMA - BB, legend = ["SMA - BB"], nf = 0)


##################################################################
############# ELABORATE PLOTS #######################################
######################################################################
        
def plot_TD(self,start_date = [], end_date = []):
    
    labels = ['Close value', "Days", 'Adjusted Close']
    
    ### GET THE SUBSELECTION OF DAYS DATA ###
    if (start_date != []) and (end_date != []):  # If we specify a date
    
        start_date = dt.datetime(start_date).astype(datetime)
        end_date = dt.datetime(end_date).astype(datetime)
 
        index_time_list = self.TD.index.date  # Obtain the date only
#            print index_time_list
        
        indexs = np.where( index_time_list < end_date) and np.where( index_time_list > start_date)
        
#            print indexs
        
        # Subselection of the indexes 
        Close = self.intraData.ix[indexs]["Close"].values
        
    else:
        Close = self.intraData["Close"].values
        
    gr.plot_graph([],Close.T,labels, 1)
