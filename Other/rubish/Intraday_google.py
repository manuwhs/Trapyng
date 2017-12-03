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
LOAD = 1;
if (LOAD == 1):
    # ITX BKIA SAN GLD
    data_google = gdl.get_google_data('GLD', 60, 30) # IB
    
    close_google = data_google["c"].values
    date_google = data_google.index.values
    date_google_normalized = itd.transform_time(date_google) 
    
    Price_days, Hour_days, days = itd.separate_days(close_google, date_google_normalized)

    # Remove days with little data
    Nsmin = 100
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

""" REMOVE -1 time intervals """
Not_trading = np.where(Price_days_mean == -1)[0]
Trading = np.where(Price_days_mean != -1)[0]

Price_days_norm = Price_days_norm[:,Trading]
Price_days_mean = Price_days_mean[Trading]

Price_days_mean2 = np.mean(Price_days_norm.T-Price_days_norm.T[0] , 1)


labels = ['Adjusted Close', "Days", 'Adjusted Close']

#gr.plot_graph([],Price_days_norm.T,labels, 1)

#gr.plot_graph([],Price_days_mean.T,labels, 1)
#
gr.plot_graph([],Price_days_norm.T - Price_days_norm.T[0],labels, 1)

#gr.plot_graph([],Price_days_mean2,labels, 1)

gr.plot_graph([],Price_days_norm.flatten(),labels, 1)
#

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
cumreturns_days = ul.get_CumReturn(Price_days_norm)
ave_cumret = np.mean(cumreturns_days,0)
std_cumret =  np.std(cumreturns_days,0)

gr.plot_graph([],cumreturns_days.T,labels, 1)

gr.plot_graph([],ave_cumret.T ,labels, 1)
gr.plot_graph([],ave_cumret.T + std_cumret.T,labels, 0)
gr.plot_graph([],ave_cumret.T - std_cumret.T,labels, 0)

""" Get Readjustments difference """ 
# We obtain for every day the difference between the open of this day
# and the close of the previous.
# Maybe if they start higher the usually go down (unless we are in a possitive trend maybe)
close_open_diff = itd.get_close_open_diff(Price_days_norm)
open_close_diff = itd.get_open_close_diff(Price_days_norm)
open_close_diff = open_close_diff[1:]  # Eliminate the first day coz we dont have the close_open

gr.scatter_graph( close_open_diff,open_close_diff ,labels, 1)

cumreturns_days = ul.get_CumReturn(Price_days_norm)
ave_cumret = np.mean(cumreturns_days,0)
std_cumret =  np.std(cumreturns_days,0)

gr.plot_graph([],cumreturns_days.T,labels, 1)

gr.plot_graph([],ave_cumret.T ,labels, 1)
gr.plot_graph([],ave_cumret.T + std_cumret.T,labels, 0)
gr.plot_graph([],ave_cumret.T - std_cumret.T,labels, 0)


# Plot derivative of the cum_std
#L = 10
#
#deriv_com_std = std_cumret[1:] - std_cumret[:-1]
#deriv_com_std = indl.simpleMean(deriv_com_std, L)
#gr.plot_graph([],cumreturns_days.T,labels, 1)
#
#gr.plot_graph([],deriv_com_std.T ,labels, 1)
# MTBA  ITX
