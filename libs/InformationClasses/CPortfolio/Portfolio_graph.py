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
