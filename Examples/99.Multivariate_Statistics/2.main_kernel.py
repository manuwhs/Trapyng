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

distribution_graph = 1
distribution_graph3D = 1
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
edate = dt.datetime.strptime("11-8-2017", "%d-%m-%Y")
#edate = dt.datetime.now()

Cartera.set_interval(sdate, edate)

opentime, closetime = Cartera.get_timeData(symbolIDs[0],15).guess_openMarketTime()
dataTransform = ["intraday", opentime, closetime]

#Cartera.get_timeData(symbolIDs[0],15).fill_data()
#Cartera.get_timeData(symbolIDs[1],15).fill_data()
ret1 = Cartera.get_timeData(symbolIDs[0],15).get_timeSeriesReturn()*100
ret2 = Cartera.get_timeData(symbolIDs[1],15).get_timeSeriesReturn()*100
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

    n_grids = 40
    kde_K = 3
    gl.init_figure()
    ax1 = gl.subplot2grid((4,4), (1,0), rowspan=3, colspan=3)
    gl.scatter(ret1,ret2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["",symbolIDs[0], symbolIDs[1]],
               legend = ["%i points"%ret1.size])
    
    ## X distribution
    ax2 = gl.subplot2grid((4,4), (0,0), rowspan=1, colspan=3, sharex = ax1)

    gl.histogram(X = ret1, ax = ax2, AxesStyle = "Normal - No xaxis", 
                 color = "k", alpha = 0.5)
                 
    x_grid = np.linspace(min(ret1),max(ret1),n_grids)
    y_val = bMA.kde_sklearn(ret1, x_grid, bandwidth=np.std(ret1)/kde_K)  
    gl.plot(x_grid, y_val, color = "k",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mean[0], cov[0,0])])
    
    # Y distribution
    ax3 = gl.subplot2grid((4,4), (1,3), rowspan=3, colspan=1,sharey = ax1,)
    gl.histogram(X = ret2, ax = ax3, orientation = "horizontal", 
                 AxesStyle = "Normal - No yaxis", 
                 color = "k", alpha = 0.5)
    x_grid = np.linspace(min(ret2),max(ret2),n_grids)
    y_val = bMA.kde_sklearn(ret2, x_grid, bandwidth=np.std(ret2)/kde_K) 
    
    gl.plot(y_val, x_grid, color = "k",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mean[0], cov[1,1])])
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    xx, yy, zz = bMA.kde2D(ret1,ret2, bandwidth = np.std(ret1)/kde_K,
                           xbins=n_grids*1j, ybins=n_grids*1j)
    ax1.contour(xx, yy, zz)
    gl.savefig(folder_images +'KDEHistogramCLOSE.png', 
               dpi = 100, sizeInches = [18, 14])
               
if(distribution_graph3D):
    ################ Contour plot of the scatter plot ################
    xx, yy, zz = bMA.kde2D(ret1,ret2, bandwidth = np.std(ret1)/kde_K,
                           xbins=n_grids*1j, ybins=n_grids*1j)
    ## Plot the 3D surface
    ax3D = gl.plot_3D(xx, yy, zz, nf = 1)
    
    ## Limits of the plotting !
    xmin,xmax = [np.min(xx.flatten()), np.max(xx.flatten())]
    ymin,ymax = [np.min(yy.flatten()), np.max(yy.flatten())]
    zmin,zmax = [np.min(zz.flatten()), np.max(zz.flatten())]
    
    # Plot the marginalization of X
    x_grid = np.linspace(min(ret1),max(ret1),n_grids)
    y_val = bMA.kde_sklearn(ret1, x_grid, bandwidth=np.std(ret1)/kde_K)  
    ax3D.plot(x_grid,y_val,ymax, zdir='y')
    ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),ymax, zdir='y')
  
    # Plot the marginalization of Y
    x_grid = np.linspace(min(ret2),max(ret2),n_grids)
    y_val = bMA.kde_sklearn(ret2, x_grid, bandwidth=np.std(ret2)/kde_K)  
    
    ax3D.plot(x_grid,y_val, xmin, zdir='x')
    ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),xmin, zdir='x')
    
    # Plot the contour lines:
    ax3D.contour(xx, yy, zz, offset=0, zdir='z')

    # Set the visualization limits !
    ax3D.set_xlim(xmin, xmax)
    ax3D.set_ylim(ymin, ymax)
    ax3D.set_zlim(zmin, zmax)
    
#    ax1.pcolormesh(xx, yy, zz)
#    ax1.imshow(zz, origin='lower', extent=[-3,3,-3,3], cmap="gray")

    gl.savefig(folder_images +'KDEHistogramCLOSE3D.png', 
               dpi = 100, sizeInches = [18, 14])
               