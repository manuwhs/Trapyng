import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from numpy import loadtxt
import time

import graph_lib as gr
import Intraday_lib as itd
import utilities_lib as ul
import indicators_lib as indl
import get_data_lib as gdl 

plt.close("all")
LOAD = 0;
if (LOAD == 1):
    file_dir = 'GBPUSD1m.txt'
    date,bid,ask = np.loadtxt(file_dir, unpack=True,
                                  delimiter=',')

#    price = (bid + ask)/2
    price = (bid + ask)/2
    Price_days, Hour_days, days = itd.separate_days(price, date)

    # Remove days with little data
    Nsmin = 10000
    Ndays = len(days)
    
    to_delete_indx = []
    for i in range(Ndays):
        if (len(Price_days[i]) < Nsmin):
            print "Day " + str(days[i]) + " removed" 
            to_delete_indx.append(i)
            
    if(len(to_delete_indx) > 0):
        Price_days = itd.remove_list_indxs(Price_days, to_delete_indx)
        Hour_days = itd.remove_list_indxs(Hour_days, to_delete_indx)
        days = itd.remove_list_indxs(days, to_delete_indx)

Ndays = len(days)

""" Average every day """
Price_days_norm = []
for day_i in range(Ndays):
    price_day = itd.time_normalizer (Hour_days[day_i], Price_days[day_i], 600)
    Price_days_norm.append(price_day)

Price_days_norm = np.array(Price_days_norm)
Price_days_mean = np.mean(Price_days_norm, 0)
Price_days_mean2 = np.mean(Price_days_norm.T-Price_days_norm.T[0] , 1)

labels = ['Adjusted Close', "Days", 'Adjusted Close']
#gr.plot_graph([],price_day,labels, 1)
#
#gr.plot_graph([],Price_days[day_i],labels, 1)

#gr.plot_graph([],Price_days_norm.T,labels, 1)
#
#gr.plot_graph([],Price_days_mean.T,labels, 1)
##
#gr.plot_graph([],Price_days_norm.T - Price_days_norm.T[0],labels, 1)
#
#gr.plot_graph([],Price_days_mean2,labels, 1)

#gr.plot_graph([],Price_days_norm.flatten(),labels, 1)


""" RETURNS """
#returns_days = ul.get_Return(Price_days_norm)
#ave_ret = np.mean(returns_days,0)
#std_ret = np.std(returns_days,0)
#
#gr.plot_graph([],returns_days.T,labels, 1)
#
#gr.plot_graph([],ave_ret.T,labels, 1)
#gr.plot_graph([],ave_ret.T + std_ret.T,labels, 0)
#gr.plot_graph([],ave_ret.T - std_ret.T,labels, 0)


""" Cummulative Returns""" 
#cumreturns_days = ul.get_CumReturn(Price_days_norm)
#ave_cumret = np.mean(cumreturns_days,0)
#std_cumret =  np.std(cumreturns_days,0)
#
#gr.plot_graph([],cumreturns_days.T,labels, 1)
#
#gr.plot_graph([],ave_cumret.T ,labels, 1)
#gr.plot_graph([],ave_cumret.T + std_cumret.T,labels, 0)
#gr.plot_graph([],ave_cumret.T - std_cumret.T,labels, 0)


# Plot derivative of the cum_std
#L = 10
#
#deriv_com_std = std_cumret[1:] - std_cumret[:-1]
#deriv_com_std = indl.simpleMean(deriv_com_std, L)
#gr.plot_graph([],cumreturns_days.T,labels, 1)
#
#gr.plot_graph([],deriv_com_std.T ,labels, 1)
# MTBA  ITX
#data_google = gdl.get_google_data('BKIA', 60, 30) # IB
#close_google = data_google["c"].values
#date_google = data_google["ts"].values
#date_google_normalized = itd.transform_time(date_google) 
#
#gr.plot_graph([],close_google ,labels, 1)

