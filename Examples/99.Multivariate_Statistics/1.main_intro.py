"""
In this document we download the 15M data for 2 companies and plot it
"""

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

##############################################
########## FLAGS ############################
trading_graph = 1
distribution_graph = 1
ellipse_graph = 1

##########################################################################
################# DATA OBTAINING ######################################
##########################################################################
######## SELECT SOURCE ########
dataSource =  "Google"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)
folder_images = "../pics/Trapying/MultivariateStat/"
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
    
sdate = dt.datetime.strptime("6-8-2017", "%d-%m-%Y")
edate = dt.datetime.strptime("15-8-2017", "%d-%m-%Y")
#edate = dt.datetime.now()

Cartera.set_interval(sdate, edate)

opentime, closetime = Cartera.get_timeData(symbolIDs[0],15).guess_openMarketTime()
dataTransform = ["intraday", opentime, closetime]

#Cartera.get_timeData(symbolIDs[0],15).fill_data()
#Cartera.get_timeData(symbolIDs[1],15).fill_data()
ret1 = Cartera.get_timeData(symbolIDs[0],15).get_timeSeriesReturn()*100
ret2 = Cartera.get_timeData(symbolIDs[1],15).get_timeSeriesReturn()*100
dates = Cartera.get_timeData(symbolIDs[1],15).get_dates()

print dates[0], dates[26], dates[27]
################# Plotting the data #################
if(trading_graph):
    # Trading plot of the points !
    gl.set_subplots(2,1)
    title = "Price evolution for 2 securities"
    ax1 = gl.tradingBarChart(Cartera.get_timeData(symbolIDs[0],15), nf = 1,
                             dataTransform = dataTransform, AxesStyle = "Normal - No xaxis",
                             labels = [title,"",symbolIDs[0] +"(15M)"])
    gl.tradingBarChart(Cartera.get_timeData(symbolIDs[1],15), nf = 1, sharex = ax1,
                       dataTransform = dataTransform, AxesStyle = "Normal",
                        labels = ["","",symbolIDs[1] +"(15M)"])

    gl.savefig(folder_images +'PriceEvolution2Symb15.png', 
               dpi = 100, sizeInches = [14, 7])
               
    # Returns stem plot of the points
    gl.set_subplots(2,1)
    title = "Return for 2 securities"
    ax1 = gl.stem(dates, ret1, nf = 1, dataTransform = dataTransform,
                   AxesStyle = "Normal - No xaxis",
                   labels = [title,"",symbolIDs[0] +"(15M)"])
    gl.stem(dates, ret2, nf = 1, dataTransform = dataTransform, sharex = ax1,
            AxesStyle = "Normal",
            labels = ["","",symbolIDs[1] +"(15M)"])
            
    gl.savefig(folder_images +'Returns2Symb15.png', 
               dpi = 100, sizeInches = [14, 7])
    
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

#### Apply some shifting if we want crosscorrelations though time
if(0):
    ret2 = ret2[1:,:]
    ret1 = ret1[:-1,:]
## Final data
dates = dates[NonNan_index[:,0]]
data = np.concatenate((ret1,ret2),axis = 1)
mean = np.mean(data, axis = 0)
corr = bMA.get_corrMatrix(data)
cov = bMA.get_covMatrix(data)
############################################################
################# PLOT DATA ###############################
############################################################

if(distribution_graph):
    # Get the histogram and gaussian estimations !
    ## Scatter plot of the points 
    gl.init_figure()
    ax1 = gl.subplot2grid((4,4), (1,0), rowspan=3, colspan=3)
    gl.scatter(ret1,ret2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["",symbolIDs[0], symbolIDs[1]],
               legend = ["%i points"%ret1.size])
    
    ## X distribution
    ax2 = gl.subplot2grid((4,4), (0,0), rowspan=1, colspan=3, sharex = ax1)
    gl.histogram(X = ret1, ax = ax2, AxesStyle = "Normal - No xaxis", 
                 color = "k", alpha = 0.5)
    
    x_grid, y_val = bMA.gaussian1D_points(X = ret1, std_K = 3)
    gl.plot(x_grid, y_val, color = "k",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mean[0], cov[0,0])])
    
    # Y distribution
    ax3 = gl.subplot2grid((4,4), (1,3), rowspan=3, colspan=1,sharey = ax1,)
    gl.histogram(X = ret2, ax = ax3, orientation = "horizontal", 
                 AxesStyle = "Normal - No yaxis", 
                 color = "k", alpha = 0.5)
                 
    x_grid, y_val = bMA.gaussian1D_points(X = ret2, std_K = 3)
    gl.plot(y_val, x_grid, color = "k",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mean[0], cov[1,1])])
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    if(ellipse_graph):
        ################# Draw the error ellipse  #################
        mean,w,h,theta = bMA.get_gaussian_ellipse_params(data, Chi2val = 2.4477)
        vecs,vals = bMA.get_eigenVectorsAndValues(data)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2,
                 legend = ["Corr: %.2f"%(corr[0,1])])
        
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

        
    gl.savefig(folder_images +'ScatterHistogramCLOSE.png', 
               dpi = 100, sizeInches = [18, 14])