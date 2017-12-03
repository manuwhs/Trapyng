import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time as time
import CMarket as CMkt 
import copy as copy
import CPortfolio as CPfl
import CPortfolio_DDBB as Pdb

import CCAPM as CCAPM
symbol = "AAPL"

plt.close("all")

PLOT_GRAPHS = 0

symbols = ["AAPL", "GLD", "FB", "IBM"]
symbols_n = ["ITX", "BKIA", "SAN"]  
start_date = [2015, 05, 05]
end_date = [2015, 10, 06]
N_days_intra = 15


""" LOAD THE NEW DATA FROM SYMBOLS""" 

Mercados = Pdb.load_symbols(symbols)
Mercado = Mercados[2]

""" PRICES AND RETURNS """



""" VELEROS """
Mercado.plot_dailyHA()
Mercado.plot_dailyVJ()

""" Moving Averages """

Mercado.plot_MA(3)
#ATR = Mercado.get_ATR()

#Mercado.scatter_deltaDailyMagic()

#Mercado.plot_TrCrMr()

#Cartera.plot_get_cumReturns()

""" Oscillators """



""" Oscillators """


