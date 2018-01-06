import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import CSymbol as CSy
import CPortfolio as CPfl
import CStrategy as CStgy

def create_matrix(prices_list):
    lengthd  = []
    Ndays = len(prices_list)
    for i in range(Ndays):
        lengthd.append(prices_list[i].size)
    
    leng = np.max(lengthd)
    print lengthd
    matrixd = np.zeros((Ndays,leng))
    for i in range(Ndays):
        real_len = prices_list[i].size
        matrixd[i,:real_len] = prices_list[i][0,:]
    
    return matrixd
    
    
plt.close("all")
symbols = ["GBPUSD","EURUSD", "Mad.ITX","CBOT.YM","XAUUSD","XAGUSD"]
periods = [1, 5,15,60,1440]

Cartera = CPfl.Portfolio(symbols,periods)
Cartera.load_symbols_csv()

Hitler = CStgy.CStrategy(Cartera)

# Correlation Distance
## 10 seems to be a good number !! Has to do with weekly shit
## Some BBDD have 70 and other 30, the 30 can also me exploited but have to find
## why the change. 

#Hitler.KNNPrice(10,10, algo = "Correlation", Npast = -1)  # 

#Hitler.XingAverages("Mad.ITX", Ls = 30,Ll = 100)  # 

#Ls_list = [20,25,30,35,40,45]
#Ll_list = [80,90,100]

Ls_list = range(20,40)
Ll_list = range(90,110)

crosses, dates = Hitler.RobustXingAverages(symbols[0], Ls_list,Ll_list)  # 

#prices, dates = Hitler.intraDayTimePatterns(symbols[3])  # 

#matrixd = create_matrix(prices)
##matrixd = np.array(prices)
#print matrixd.shape
#
#labels = labels = ["IntraTimePatterns", "Time", "Price"]
#gr.plot_graph([],np.mean(matrixd, axis = 0),labels,new_fig = 1)
#gr.plot_graph([],np.mean(matrixd, axis = 0) + np.std(matrixd, axis = 0),labels,new_fig = 0)
#gr.plot_graph([],np.mean(matrixd, axis = 0) - np.std(matrixd, axis = 0),labels,new_fig = 0)    
#Hitler.XingAverages("Mad.ITX", Ls = 30,Ll = 100)  # 
#

    
        