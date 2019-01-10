# -*- coding: utf-8 -*-
#import matplotlib

import numpy as np

import time
import pandas as pd
import graph_lib as gr
import Intraday_lib as itd
import utilities_lib as ul
import indicators_lib as indl
import oscillators_lib as oscl
import get_data_lib as gdl 

import datetime as dt
from datetime import datetime

"""
Library with all the obtaining indicator functions of the market.

"""

###########################################
########### Moving Averages ###############
###########################################

def get_SMA(self, L ):
    timeSeries = self.get_timeSeries()
    SMA = indl.get_SMA(timeSeries, L)
    return SMA

def get_WMA(self, L ):
    timeSeries = self.get_timeSeries()
    WMA = indl.get_WMA(timeSeries, L)
    return WMA
    
def get_EMA(self, L, alpha = -1):
    timeSeries = self.get_timeSeries()
    EMA = indl.get_EMA(timeSeries, L, alpha)
    return EMA
    
def get_HMA(self, L):
    timeSeries = self.get_timeSeries()
    HMA = indl.get_HMA(timeSeries, L)
    return HMA

def get_HMAg(self, L, alpha = -1):
    timeSeries = self.get_timeSeries()
    EMA = indl.get_HMAg(timeSeries, L, alpha)
    return EMA

def get_TMA(self, L):
    timeSeries = self.get_timeSeries()
    TMA = indl.get_TMA(timeSeries, L)
    return TMA
    
def get_TrCrMr(self, alpha = -1):
    timeSeries = self.get_timeSeries()
    get_TrCrMr = indl.get_TrCrMr(timeSeries, alpha)
    return get_TrCrMr




""" Volatility Shit """
def get_ATR(self):
    """ ATR s an indicator that shows volatility of the market
    True Range is the greatest of the following three values:
            1- difference between the current maximum and minimum (high and low);
            2- difference between the previous closing price and the current maximum;
            3- difference between the previous closing price and the current minimum.
        It just wants to obtain the maximum range
    """

    """ Average True Range for volatility """
    rangeOC = self.get_timeSeries(["RangeCO"])
    magicDelta = self.get_timeSeries(["magicDelta"])
    diffPrevCloseCurrMin = self.get_diffPrevCloseCurrMin()
    diffPrevCloseCurrMax = self.get_diffPrevCloseCurrMax()
    
    print (rangeOC.shape, magicDelta.shape, diffPrevCloseCurrMin.shape, diffPrevCloseCurrMax.shape)
#    print dailyDelta.shape, magicDelta.shape, diffPrevCloseCurrMin.shape,diffPrevCloseCurrMax.shape
    All_diff = np.concatenate((rangeOC,magicDelta,diffPrevCloseCurrMin,diffPrevCloseCurrMax), axis = 1)
    All_diff = np.abs(All_diff)
    
    ATR = np.max(All_diff, axis = 1)
    
    return ATR

def get_MACD(self, Ls = 12, Ll = 26, Lsmoth = 9, alpha = -1):
    MACD = oscl.get_MACD(self.get_timeSeries(),Ls, Ll, Lsmoth,alpha)
    return MACD
    
def get_momentum(self, N = 1):
    momentum = oscl.get_momentum(self.get_timeSeries(["Close"]), N)
    return momentum

def get_RSI(self, N = 1):
    RSI = oscl.get_RSI(self.get_timeSeries(["Close"]), N)
    return RSI

def get_stochasticOscillator(self, N = 1):
    momentum = indl.get_momentum(self.get_timeSeries(["Close"]), N)
    return momentum





def get_drawdown(self):  # TODO
    """
    calculate max drawdown and duration
 
    Returns:
        drawdown : vector of drawdwon values
        duration : vector of drawdown duration
    """
    cumret = pnl
 
    highwatermark = [0]
 
    idx = pnl.index
    drawdown = pd.Series(index = idx)
    drawdowndur = pd.Series(index = idx)
 
    for t in range(1, len(idx)) :
        highwatermark.append(max(highwatermark[t-1], cumret[t]))
        drawdown[t]= (highwatermark[t]-cumret[t])
        drawdowndur[t]= (0 if drawdown[t] == 0 else drawdowndur[t-1]+1)
 
    return drawdown, drawdowndur

def get_BollingerBand(self, L):
    timeSeries = self.get_timeSeries()
    BB = voll.get_BollingerBand(timeSeries, L)
    return BB
    
    