
"""
ValueError: DateFormatter found a value of x=0, which is an illegal date.  This usually occurs because you have not informed the axis that it is plotting dates, e.g., with ax.xaxis_date()
"""
import os
os.chdir("../")
import import_folders

import utilities_lib as ul

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import CSymbol as CSy
import copy as copy
import CPortfolio as CPfl
#import CStrategy as CStgy
from graph_lib import gl 

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
import pandas as pd
from sklearn import linear_model
import basicMathlib as bMA
import indicators_lib as indl

plt.close("all")
######## SELECT DATASET, SYMBOLS AND PERIODS ########
# Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = "GCI")

################## Date info ###################
sdate_str = "01-8-2014"
edate_str = "21-12-2016"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

####################################################
#### Load the info about the available symbols #####
####################################################
load_info = 0
if (load_info == 1):
    Symbol_info = CSy.load_symbols_info(info_folder)
    Symbol_names = Symbol_info["Symbol_name"].tolist()
    Nsym = len(Symbol_names)

####################################################
####### SELECT SYMBOLS AND PERIODS #################

symbols = ["Alcoa_Inc", "Apple_Comp.","Ford_Motor", "Amazon"]

periods = [1440]
period = periods[0]

####################################################
####### LOAD SYMBOLS AND SET    ###################
####################################################
Cartera = CPfl.Portfolio(symbols, periods)   # Set the symbols and periods to load
# Download if needed.
#Cartera.update_symbols_csv_yahoo(sdate_str,edate_str,storage_folder)    # Load the symbols and periods
Cartera.set_csv(storage_folder)    # Load the symbols and periods

## SET THINGS FOR ALL OF THEM
Cartera.set_interval(sdate,edate)
Cartera.set_seriesNames(["Close"])

############################################################
######################## FILLING DATA  ##########################
############################################################

print ("Filling Data")
Cartera.fill_data()
print ("Data Filled")

prices = Cartera.get_timeSeries(symbolIDs = [symbols[1]])
prices = Cartera.get_timeSeries()
dates = Cartera.get_dates(period, Cartera.symbol_names[0])

# We get now the moving averages
indx = 1
lag = 35

## MACD Parameters !!
n_fast = 12; n_slow = 26; n_smooth = 9
EMAfast = Cartera.EMA(n = n_fast)
EMAslow = Cartera.EMA(n = n_slow)
MACD = EMAfast - EMAslow
MACD = pd.ewma(MACD, span = n_smooth, min_periods = n_smooth - 1)

# Smoothin of velocity, we want to enter when it is falling
MACD_vel = bMA.diff(MACD,n = 1)
nsmooth_vel = 4
MACD_vel = indl.get_SMA(MACD_vel, L = nsmooth_vel)

## RSI Parameter !
RSI = Cartera.RSI(n = 14)
RSI_vel = bMA.diff(RSI,n = 1)
#RSI_vel = indl.get_SMA(RSI_vel, L = nsmooth_vel)

## ATR Parameter !
ATR = Cartera.ATR(n = 14)
ATR_vel = bMA.diff(ATR,n = 1)
RSI_vel = indl.get_SMA(ATR_vel, L = nsmooth_vel)

###########################################################
################# PREPARE THE DATA ########################
###########################################################
X_data = np.concatenate((MACD[:,[indx]],MACD_vel[:,[indx]]), axis = 1)
X_data = np.concatenate((X_data,RSI[:,[indx]], ATR[:,[indx]]), axis = 1)
X_data = np.concatenate((X_data,RSI_vel[:,[indx]], ATR_vel[:,[indx]]), axis = 1)

Y_data = bMA.diff(prices[:,indx], lag = lag, cval = np.NaN)
Y_data = bMA.shift(Y_data, lag = -lag, cval = np.NaN)


### Returns 
lag_ret = 20
return_Ydata = bMA.get_return(prices[:,[indx]], lag = lag_ret)
reconstruct_Ydata = bMA.reconstruc_return(prices[:,[indx]], return_Ydata, lag = lag_ret)

gl.plot([],prices[:,[indx]], legend= ["price"])
gl.plot([],reconstruct_Ydata, nf = 0, legend= ["reconstruction"])
gl.plot([],return_Ydata, nf = 0, na = 1, legend= ["return"])

Y_data = return_Ydata
Y_data = bMA.shift(Y_data, lag = -lag, cval = np.NaN)


def filter_by_risk():
    # This funciton will filter the samples used in the analysis 
    # 
    pass
    # We also should analyse abs(ret) pare detectar que cuando hay mucho riesgo
    # despues hay una tendancia clara.
    
condition_data = 0
if (condition_data):

    ## Get the selection
    mask_sel = np.argwhere(abs(X_data[:,0]) > 0.2)[:,0]
    #mask_sel_neg = np.argwhere(X_data[:,0] > -4)[:,0]
    ## Replace by Nans the others
    X_data_aux = np.ones(X_data.shape) * np.nan 
    Y_data_aux = np.ones(Y_data.shape) * np.nan 
    #X_data = X_data[mask_sel,:]
    #Y_data = Y_data[mask_sel,:]
    X_data_aux[mask_sel,:] = X_data[mask_sel,:]
    Y_data_aux[mask_sel,:] = Y_data[mask_sel,:]
    
    X_data = X_data_aux
    Y_data = Y_data_aux
    
    ## Redo another selection
    mask_sel = np.argwhere(abs(X_data[:,1]) < 0.1)[:,0]
    #mask_sel_neg = np.argwhere(X_data[:,0] > -4)[:,0]
    ## Replace by Nans the others
    X_data_aux = np.ones(X_data.shape) * np.nan 
    Y_data_aux = np.ones(Y_data.shape) * np.nan 
    #X_data = X_data[mask_sel,:]
    #Y_data = Y_data[mask_sel,:]
    X_data_aux[mask_sel,:] = X_data[mask_sel,:]
    Y_data_aux[mask_sel,:] = Y_data[mask_sel,:]
    
    X_data = X_data_aux
    Y_data = Y_data_aux

#Y_data = np.sign(Y_data)
#Y_data = bMA.shift(prices[:,indx], lag = 5, cval = np.NaN)
# We can also have the data in dataframe form
########################################################################
################## Write the data in pandas format ###############
########################################################################
data = pd.DataFrame(
{'MACD': MACD[:,indx], 'MACD_vel':  MACD_vel[:,indx], 'ATR' : ATR[:,indx], 
'RSI': RSI[:,indx], 'RSI_vel': RSI_vel[:,indx], 'ATR_vel': ATR_vel[:,indx], 'Y':  Y_data[:,0]})

################################################################
###################  PLOTTING THE DATA #########################
################################################################

## Plotting plotting
plotting_some = 1
if (plotting_some == 1):
    Ndiv = 6; HPV = 2
    ### PLOT THE ORIGINAL PRICE AND MOVING AVERAGES
    timeData = Cartera.get_timeDataObj(period = -1, symbol_names = [symbols[indx]])[0]
    
    gl.init_figure()
    ##########################  AX0 ########################
    ### Plot the initial Price Volume
    gl.subplot2grid((Ndiv,4), (0,0), rowspan=HPV, colspan=4)
#    gl.plot_indicator(timeData, Ndiv = Ndiv, HPV = HPV)
    gl.plot(dates, prices[:,indx], legend = ["Price"], nf = 0,
            labels = ["Xing Average Strategy","Price","Time"])
    
    Volume = timeData.get_timeSeries(seriesNames = ["Volume"])
    gl.plot(dates, Volume, nf = 0, na = 1, lw = 0, alpha = 0)
    gl.fill_between(dates, Volume)
    
    axes = gl.get_axes()
    axP = axes[0]  # Price axes
    axV = axes[1]  # Volumne exes
    
    axV.set_ylim(0,3 * max(Volume))
    gl.plot(dates, EMAfast[:,indx],ax = axP, legend = ["EMA = %i" %(n_fast)], nf = 0)
    gl.plot(dates, EMAslow[:,indx],ax = axP, legend = ["EMA = %i" %(n_slow)], nf = 0)
    
    ##########################  AX1 ########################
    pos = 0
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    gl.plot(dates, Y_data, nf = 0, legend = ["Y data filtered"])
    
   ##########################  AX2 ########################
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    
    gl.plot(dates, MACD[:,indx], nf = 0, legend = ["MACD"])
    gl.plot(dates, MACD_vel[:,indx], nf = 0, na = 1, legend = ["MACD_vel"])
    
    ##########################  AX3 ########################
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)
    gl.plot(dates, RSI[:,indx], nf = 0, legend = [ "RSI"])
    gl.plot(dates, RSI_vel[:,indx], nf = 0, na = 1, legend = [ "RSI_vel"])
    ##########################  AX4 ########################
    pos = pos +1
    gl.subplot2grid((Ndiv,4), (HPV + pos,0), rowspan=1, colspan=4, sharex = axP)

#    BB = timeData.BBANDS(seriesNames = ["Close"], n = 14)
#    gl.plot(dates,  BB[:,0] - BB[:,1] , nf = 0, na = 0, legend = [ "BB"])
    gl.plot(dates, ATR[:,indx], nf = 0,legend = ["ATR"])
    gl.plot(dates, ATR_vel[:,indx], nf = 0, na = 1,legend = ["ATR_vel"])

    ## Format the axis
    all_axes = gl.get_axes()
    for i in range(len(all_axes)-2):
        ax = all_axes[i]
        plt.setp(ax.get_xticklabels(), visible=False)
    
    gl.format_axis2(all_axes[-2], Nx = 20, Ny = 5, fontsize = -1, rotation = 45)
    plt.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0)

model_OLS = 1
if (model_OLS):
    ##############################################################################
    # Multilinear regression model, calculating fit, P-values, confidence
    # intervals etc.
    # Fit the model
    model = ols("Y ~ MACD + RSI + ATR + MACD_vel  + ATR_vel + RSI_vel", data).fit()
    params = model._results.params
    # Print the summary
    print(model.summary())
    
    print("OLS model Parameters")
    print(params)

    Xdata = np.array([MACD[:,indx], RSI[:,indx], MACD_vel[:,indx]]).T
    ## Graoca bonita !! TODO
    print (ul.fnp(np.sqrt(np.sum(Xdata*Xdata, axis = 1))).shape)
    Xdata = Xdata / ul.fnp(np.sqrt(np.sum(Xdata*Xdata, axis = 1)))
    gl.scatter_3D(Xdata[:,0],Xdata[:,1],Xdata[:,2])
    
    gl.scatter_3D(-Xdata[:,0],-Xdata[:,1],-Xdata[:,2], nf = 0)
    gl.scatter_3D(-Xdata[:,0],Xdata[:,1],-Xdata[:,2], nf = 0)
    gl.scatter_3D(Xdata[:,0],-Xdata[:,1],Xdata[:,2], nf = 0)
    # Peform analysis of variance on fitted linear model
    #anova_results = anova_lm(model)
    #print('\nANOVA results')
    #print(anova_results)

##############################################################################
model_sklearn = 1
if (model_sklearn):

    ## Eliminate the Nans !! Even if they appear in just one dim
    mask_X = np.sum(np.isnan(X_data), axis = 1) == 0
    mask_Y = np.isnan(Y_data) == 0
    mask = mask_X & mask_Y[:,0]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_data[mask,:], Y_data[mask,:])
    
    #    coeffs = np.array([regr.intercept_, regr.coef_])[0]
    coeffs = np.append(regr.intercept_, regr.coef_)
    params = np.array(coeffs)

    residual = regr.residues_
    print("sklearn model Parameters")
    print(params)
    print("Residual")
    print (residual)


def get_Residuals(X_data, prices, lag = 20):
    ## Prediction of the thingies
    Y_data = bMA.diff(prices, lag = lag, cval = np.NaN)
    Y_data = bMA.shift(Y_data, lag = -lag, cval = np.NaN)

    ## Eliminate the Nans !! Even if they appear in just one dim
    mask_X = np.sum(np.isnan(X_data), axis = 1) == 0
    mask_Y = np.isnan(Y_data) == 0
    mask = mask_X & mask_Y[:,0]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_data[mask,:], Y_data[mask,:])
    
    #    coeffs = np.array([regr.intercept_, regr.coef_])[0]
    coeffs = np.append(regr.intercept_, regr.coef_)
    params = np.array(coeffs)

    residual = regr.residues_
    residual = regr.score(X_data[mask,:], Y_data[mask,:])
    return residual

lags = range(1,100)

residuals = []
for lag in lags:
    residuals.append(get_Residuals(X_data,prices[:,indx], lag = lag))

gl.plot(lags, residuals)
##########################################################
# Plotting of the regression
plotting_3D = 1
if (plotting_3D):
#    Y_data = np.sign(Y_data)
    gl.scatter_3D(X_data[:,0],X_data[:,1], Y_data,
                   labels = ["","",""],
                   legend = ["Pene"],
                   nf = 1)
                   
    grids = ul.get_grids(X_data)
    z = bMA.get_plane_Z(grids[0], grids[1],params)
    h = gl.plot_3D(grids[0],grids[1],z, nf = 0)

###################################
##### TODO ##################3
##################################

# How it would be to take reconstruct from the diff signal

# We shift to the right to that 
# v[i] = x[i] - x[i-1] and x[i]
# If we do v[j] = v[i+1] = x[i+1] - x[i]
# Then v[j] + x[i] = x[i+1]
# So then we shift to the left

#vel_shifted = np.roll(MACD_vel[:,[indx]],-1, axis = 0)
#reconstruction = vel_shifted + ul.fnp(MACD[:,indx])
#reconstruction = np.roll(reconstruction,1, axis = 0)
#gl.plot(dates, reconstruction, nf = 0, na = 0)











#Hitler = CStgy.CStrategy(Cartera)

# Correlation Distance
## 10 seems to be a good number !! Has to do with weekly shit
## Some BBDD have 70 and other 30, the 30 can also me exploited but have to find
## why the change. 

#Hitler.KNNPrice(10,10, algo = "Correlation", Npast = -1)  # 

#Hitler.XingAverages("Mad.ITX", Ls = 30,Ll = 100)  # 

#Ls_list = [20,25,30,35,40,45]
#Ll_list = [80,90,100]

#Ls_list = range(20,40)
#Ll_list = range(90,110)
#
#crosses, dates = Hitler.RobustXingAverages(symbols[0], Ls_list,Ll_list)  # 

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

    
        