
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utilities_lib as ul
import graph_lib as gr

import datetime as dt

def intraDayTimePatterns(self, symbol):
    # L: Length of the pattern
    # K: Number of most simmilar patterns
    # Npast: Number of past patterns to use.

    # Uses the Knn of the K most similar values for a pattern of length L
    # Simmilarity is equal 

    timeData = self.Portfolio.symbols[symbol].TDs[60]
    timeData.get_timeSeries(["Close"])
    prices, dates = timeData.get_intra_by_days(["Close"])

    labels = labels = ["IntraTimePatterns", "Time", "Price"]
    Ndays = len(prices)
    
    for i in range(Ndays):
        prices[i] = ul.get_cumReturn(prices[i])
        
    gr.plot_graph([],prices[0].T,labels,new_fig = 1)
    
    for i in range(Ndays):
        gr.plot_graph([],prices[i].T,labels,new_fig = 0)
    
    Nsuccess = 0;
    
    n_medium = 1;
    
    for i in range(Ndays):
#        print prices[i].shape
        if (prices[i].size < 7):
            continue;
            
        if (prices[i][0,n_medium] * (prices[i][0,-1] - prices[i][0,n_medium]) > 0):
#        if ((prices[i][0,4] * prices[i][0,-1]) > 0):
#        if ((prices[i][0,0] - prices[i][0,-1]) > 0):
            # If they have the same sign
            Nsuccess += 1;
    
    Nsuccess = float(Nsuccess)/Ndays
    
    print Nsuccess
    print Ndays 
    
    return prices, dates
    
    
    