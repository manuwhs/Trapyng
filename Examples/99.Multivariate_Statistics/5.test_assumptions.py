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
from scipy.stats import t
from scipy.stats import chi2

plt.close("all")

##############################################
########## FLAGS ############################

qq_plot = 1

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
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

cdf = stats.norm.cdf
data = ret1;


if(qq_plot):
    # Get the histogram and gaussian estimations !
    ## Scatter plot of the points 


    gl.set_subplots(1,2)
    
    x_grid_emp, y_val_emp = bMA.empirical_1D_cdf(ret1)
    
    x_grid = np.linspace(x_grid_emp[0],x_grid_emp[-1],100)
    x_grid, y_val = bMA.gaussian1D_points_cdf(X = ret1, x_grid = x_grid)
    
    ax1 = gl.plot(X = x_grid, Y = y_val, AxesStyle = "Normal", nf = 1,
                 color = "k", alpha = 0.5)
    
    gl.scatter(x_grid_emp, y_val_emp, color = "k",
            labels = ["","",""], legend = ["empirical cdf"])
    
    
    ## Now here we just plot it one againts each other like in a regression 
    ## problem !
    
#    ax2 = gl.plot(X = y_val_emp, Y = y_val, AxesStyle = "Normal", nf = 1,
#                 color = "k", alpha = 0.5)
    x_grid, y_val = bMA.gaussian1D_points_cdf(X = ret1, x_grid = x_grid_emp)
    gl.scatter(X = y_val_emp, Y = y_val, color = "k",
            labels = ["","",""], legend = ["empirical cdf"], nf = 1)
            
    gl.savefig(folder_images +'q-qplot.png', 
               dpi = 100, sizeInches = [14,6])