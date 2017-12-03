import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
# Data Structures Data
import CPortfolio as CPfl
import CSymbol as CSy
# Own graphical library
from graph_lib import gl 
# Import functions independent of DataStructure
import utilities_lib as ul
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
import pandas as pd
from sklearn import linear_model
import basicMathlib as bMA
import indicators_lib as indl
import DDBB_lib as DBl
plt.close("all")

##########################################################################
################# DATA OBTAINING ######################################
##########################################################################

######## SELECT SOURCE ########
dataSource =  "Google"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)
folder_images = "../pics/gl/"
######## SELECT SYMBOLS AND PERIODS ########
symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
symbols = ["GooG", "Alcoa_Inc"]
periods = [15]

# Create porfolio and load data
cmplist = DBl.read_NASDAQ_companies(whole_path = "../storage/Google/companylist.csv")
cmplist.sort_values(by = ['MarketCap'], ascending=[0],inplace = True)
symbolIDs = cmplist.iloc[0:3]["Symbol"].tolist()
for period in periods:
    Cartera = CPfl.Portfolio("BEST_PF", symbolIDs, [period]) 
    Cartera.set_csv(storage_folder)
    
sdate = dt.datetime.strptime("25-7-2017", "%d-%m-%Y")
#edate = dt.datetime.strptime("25-11-2016", "%d-%m-%Y")
edate = dt.datetime.strptime("25-8-2017", "%d-%m-%Y")

Cartera.set_interval(sdate, edate)

opentime, closetime = Cartera.get_timeData(symbolIDs[0],15).guess_openMarketTime()
dataTransform = ["intraday", opentime, closetime]

#Cartera.get_timeData(symbolIDs[0],15).fill_data()
#Cartera.get_timeData(symbolIDs[1],15).fill_data()
ret1 = Cartera.get_timeData(symbolIDs[0],15).get_timeSeriesReturn()
ret2 = Cartera.get_timeData(symbolIDs[1],15).get_timeSeriesReturn()
dates = Cartera.get_timeData(symbolIDs[1],15).get_dates()

##########################################################################
################# PREPROCESS DATA ######################################
##########################################################################

## Set GAP return as NAN
gap_ret = np.where(dates.time == dates[0].time())[0]
ret1[gap_ret,:] = np.NaN
ret2[gap_ret,:] = np.NaN

# Remove the NaNs
NonNan_index =  np.logical_not(np.isnan(ret1))
ret1 = ret1[NonNan_index[:,0],:]
ret2 = ret2[NonNan_index[:,0],:]
dates = dates[NonNan_index[:,0]]


data = np.concatenate((ret1,ret2),axis = 1)


## Remove samples whose module is too small.
## Can we get the optimal by Signal to Noise ratio ?

remove_central = 0
transform_circular = 1
if (remove_central):
# If the module is not big enough, we discharge it, noise
    modules = np.sqrt(np.sum(data * data,axis = 1))
    modules = modules.reshape(modules.size,1)
    module_mean = np.mean(modules)
    samples_indx = np.where(modules > module_mean*1.4)[0]
    
    print "Left %i/%i"%(samples_indx.size, modules.size)
    
    ret1 = ret1[samples_indx,:]
    ret2 = ret2[samples_indx,:]
    modules = modules[samples_indx,:]
    data = np.concatenate((ret1,ret2),axis = 1)

if (transform_circular):
## Circular transformation
    ret1[:] = ret1[:]/modules
    ret2[:] = ret2[:]/modules
    data = np.concatenate((ret1,ret2),axis = 1)
############################################################
################# PLOT DATA ###############################
############################################################
trading_graph = 0
if(trading_graph):
    # Trading plot of the points !
    gl.set_subplots(2,1)
    ax1 = gl.tradingBarChart(Cartera.get_timeData(symbolIDs[0],15), nf = 1)
    gl.tradingBarChart(Cartera.get_timeData(symbolIDs[1],15), nf = 1, sharex = ax1)
    
    # Returns stem plot of the points
    gl.set_subplots(2,1)
    ax1 = gl.stem(dates, ret1, nf = 1, dataTransform = dataTransform)
    gl.stem(dates, ret2, nf = 1, dataTransform = dataTransform, sharex = ax1)

distribution_graph = 1
ellipse_graph = 1
if(distribution_graph):
    # Get the histogram and gaussian estimations !
    ## Scatter plot of the points 
    ax1 = gl.subplot2grid((4,4), (1,0), rowspan=3, colspan=3)
    gl.scatter(ret1,ret2, alpha = 0.2, ax = ax1, AxesStyle = "Normal")
    
    ## X distribution
    ax2 = gl.subplot2grid((4,4), (0,0), rowspan=1, colspan=3, sharex = ax1)
    gl.histogram(X = ret1, ax = ax2, AxesStyle = "Normal - No xaxis", 
                 color = "k", alpha = 0.5)
    # Y distribution
    ax3 = gl.subplot2grid((4,4), (1,3), rowspan=3, colspan=1,sharey = ax1,)
    gl.histogram(X = ret2, ax = ax3, orientation = "horizontal", 
                 AxesStyle = "Normal - No yaxis", 
                 color = "k", alpha = 0.5)
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    if(ellipse_graph):
        ################# Draw the error ellipse  #################
        mean,w,h,theta = bMA.get_gaussian_ellipse_params(data, Chi2val = 2.4477)
        vecs,vals = bMA.get_eigenVectorsAndValues(data)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        ax1.plot(r_ellipse[:,0], r_ellipse[:,1], ls = "--",color = "k", lw = 2)
        
        
        gl.plot([mean[0], mean[0] + vecs[0,0]*w], 
                [mean[1], mean[1] + vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
        gl.plot([mean[0], mean[0] + vecs[1,0]*h], 
                [mean[1], mean[1] + vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
        
        gl.plot([mean[0], mean[0] - vecs[0,0]*w], 
                [mean[1], mean[1] - vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
        gl.plot([mean[0], mean[0] - vecs[1,0]*h], 
                [mean[1], mean[1] - vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
                
        ax1.axis('equal')
        gl.set_zoom(ax = ax1, X =r_ellipse[:,0], Y = r_ellipse[:,1],
                    ylimPad = [0.2,0.2],xlimPad = [0.2,0.2])


if (0):
    Ns = [1,2,5,30,300,1000]
    
    p = 0.2
    signal = np.array([0.3,0.1,0.4,0.2])
    
    
    gl.init_figure()
    gl.set_subplots(3,2)
    for N in Ns:
        total_signal = np.array([1])
        for i in range(N):
            total_signal = bMA.convolve(signal,total_signal)
    
    
        gl.stem([],total_signal, nf = 1, labels = ["N = %i"%N])