
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import functions as fu
import indicators_lib as indl
import QTSK_func as QTSK

import graph_lib as gr
plt.close("all")

PLOT_GRAPHS = 1
#plt.clf()

""" PARAMETERS TO GET THE DATA """
allocation = [0.3,0.1,0.3,0.1,0.2]
symbols = ["AAPL", "GLD", "GOOG", "$SPX", "XOM"]  
start_date = [2010, 1, 1]
end_date = [2010, 12, 31]
keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

# Get the prices of all the companies
d_data, date = QTSK.get_Prices (start_date, end_date, symbols, keys)

# Get only the close price of the desired company
security = "GLD"
GLD_close_Price = d_data["close"][security][:].values
GLD_Volume = d_data["volume"][security][:].values

#""" PLOT THE TIME SERIES DATA """
#labels = ['Adjusted Close', "Days", 'Adjusted Close', [security]]
#
#if (PLOT_GRAPHS):
#    fu.plot_graph(date,GLD_close_Price,labels, 1)
#
#""" CREATE FIGURE OF RETURNS """
#Returns = fu.get_Return(GLD_close_Price)
#Returns = np.array(Returns).T
#
#labels = ['Adjusted Close Returns', "Days", 'Adjusted Close', [security]]
#if (PLOT_GRAPHS):
#    fu.plot_graph([],Returns,labels, 1)
    
#""" CREATE FIGURE OF PRICE / VOLUME """
#labels = ['Price / Volume', "Days", 'Adjusted Close', [security + " Price", security + " Volume" ]]
#gr.price_volume_graph(date,GLD_close_Price,GLD_Volume,labels,new_fig = 1)


#""" BUILD NORMAL GRAPH """
#
#Ns = 20
#s_close = d_data["close"][security][:Ns].values
#s_open = d_data["open"][security][:Ns].values
#s_max = d_data["high"][security][:Ns].values
#s_min = d_data["low"][security][:Ns].values
#
#labels = ["Price boxes", "Days", "Price"]
#s_data = np.array([s_close,s_open,s_max,s_min])
#gr.Velero_graph(date, s_data, labels,new_fig = 1)
#gr.Heiken_Ashi_graph(date, s_data, labels,new_fig = 1)

""" Obtain MEAN indicators and Plot them """

L = 10
sM = indl.simpleMean(GLD_close_Price,L)
wM = indl.weighedMean(GLD_close_Price,L)
eM = indl.ExponentialMean(GLD_close_Price,L, alpha = 2.0/(L+1))
tM = indl.TrainedMean(GLD_close_Price,L)

TCM = indl.TCM (GLD_close_Price, alpha = -1)

gr.plot_graph(date,GLD_close_Price,labels, 1)
gr.plot_graph(date,sM,labels, 0)
gr.plot_graph(date,wM,labels, 0)
gr.plot_graph(date,eM,labels, 0)
gr.plot_graph(date,tM,labels, 0)

gr.plot_graph(date,GLD_close_Price,labels, 1)
gr.plot_graph(date,TCM,labels, 0)


