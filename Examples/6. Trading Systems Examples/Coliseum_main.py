
import import_folders
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import CSymbol as CSy
import copy as copy
import CColiseum as CCol
import CPortfolio as CPfl

plt.close("all")
PLOT_GRAPHS = 0

periods = [5,15,60,1440]
Cartera = CPfl.Portfolio(["Mad.ITX", "EURUSD", "CBOT.YM"], periods) 
Cartera.load_symbols_csv()

""" LOAD THE NEW DATA FROM SYMBOLS""" 

Coliseo = CCol.CColiseum(Portfolio = Cartera)
Coliseo.set_date(dt.datetime(2016,2,1))
Coliseo.open_position("Mad.ITX","SELL", 20)
Coliseo.open_position("Mad.ITX","SELL", 20)

Coliseo.open_position("EURUSD","BUY", 20)
Coliseo.open_position("CBOT.YM","BUY", 20)

mis = Coliseo.load_csv("../Trader/MQL4/Files/")
#ind = Coliseo.get_position_indx("GLD","BUY", 20)
#ind = Coliseo.close_position_by_indx(ind)