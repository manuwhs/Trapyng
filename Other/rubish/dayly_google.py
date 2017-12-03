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
    symbol = "SAN"
    start_date = [2010, 1, 1]
    end_date = [2010, 12, 31]
    keys = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    data_google = gdl.get_dayly_google('GLD', start_date, end_date) # IB
    
    close_google = data_google["Volume"].values
    date_google = data_google.index.values
    date_google_normalized = itd.transform_time(date_google) 

labels = ['Adjusted Close', "Days", 'Adjusted Close']

gr.plot_graph([],close_google,labels, 1)