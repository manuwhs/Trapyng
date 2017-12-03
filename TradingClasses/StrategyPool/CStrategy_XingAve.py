
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr


import datetime as dt

def check_crossing(MAs, MAl):
    Nsamples = MAs.size
    order_list = np.zeros((Nsamples,1)) # 0 = Hold, 1 = Buy, -1 = Sell
    for i in range(Nsamples):
        prev = MAs[i-1] > MAl[i-1]
        current = MAs[i] > MAl[i]
        
        if (prev != current):  # If the sign has not converted
            if (current == True): # If the short now is bigger than the long
                order_list[i] = 1;
            else:
                order_list[i] = -1;
    
    return order_list

def XingAverages(self, symbol, Ls = 10, Ll = 30):
## Crossing Average Strategy. 
## Using historical data this function detects crossing average shit and uses them as event
#    print "Using CrossAverage Strategy" 
    
    timeDataDayly = self.Portfolio.symbols[symbol].TDs[1440]
    
    dayPrice = timeDataDayly.get_timeSeries(["Close"])
    dates = timeDataDayly.get_dates()
    
    MAs = timeDataDayly.get_EMA(Ls)  # 
    MAl = timeDataDayly.get_EMA(Ll)
#    print MAs.shape, MAl.shape, dayPrice.shape
    
    np.concatenate((MAl,MAs,dayPrice[0,:].T),axis = 1)
    
    crosses = check_crossing(MAs,MAl)
    
    
    labels = labels = ["XingAve", "Time", "Price", ["MAl", "MAs", "Price"]]
    gr.plot_graph([],MAl,labels,new_fig = 1)
    gr.plot_graph([],MAs,labels,new_fig = 0)
    gr.plot_graph([],dayPrice[0,:].T,labels,new_fig = 0)
    gr.plot_graph([],MAl[0] + 0.2*(MAl[-1] - MAl[0]) *crosses/np.max(np.abs(crosses)),labels,new_fig = 0)

    
    return crosses, dates