
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_lib as gr


import datetime as dt


def RobustXingAverages(self, symbol, Ls_list = [10], Ll_list = [30]):
    # This function applies the XingAve for different values of Ls y Ll and combine them
    # To output a more robust value

    All_crosses_list = [];
    
    for Ls in Ls_list:
        for Ll in Ll_list:
            crosses, dates = self.XingAverages(symbol, Ls , Ll)
            All_crosses_list.append(crosses)
    
    labels = labels = ["XingAve", "Time", "Price", ["MAl", "MAs", "Price"]]
#    gr.plot_graph([],All_crosses_list[0],labels,new_fig = 1)
#    
#    for crosses in All_crosses_list:
#        gr.plot_graph([],crosses,labels,new_fig = 0)
    
    ######## MERGE THE INFO OF ALL CROSSES #######################
    
    final_result = np.zeros((dates.size,1))
    
    window = np.ones((1,1))[:,0]
    for crosses in All_crosses_list:
        conved = np.convolve(crosses[:,0],window)
#        print conved.shape
        final_result[:,0] += conved[:dates.size]
        
    # El buen plotting
    timeDataDayly = self.Portfolio.symbols[symbol].TDs[1440]
    dayPrice = timeDataDayly.get_timeSeries(["Close"])
    gr.plot_graph([],dayPrice[0,:].T,labels,new_fig = 1)
    gr.plot_graph([],dayPrice[0,0] + 0.5 * final_result/np.max(np.abs(final_result)),labels,new_fig = 0)
    
    return All_crosses_list, dates