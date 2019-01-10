import os
os.chdir("../")
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

plt.close("all")

################# FLAGS to activate ########################
model_OLS = 1 # Use linear model from the OLS library
model_sklearn = 1  # Use linear model from the sklearn library
lag_analysis = 1
################# READ THE DATA FROM DISK ########################
# This way we do not have to write everyhing again.
# We create the data in the previous file up to some point and
# we read it with this one.

######## PANDAS FORMAT
folder_dataFeatures = "./data/"
data = pd.read_csv(folder_dataFeatures + "dataPandas.csv", sep = ',', index_col = 0, 
                      dtype = {"Date":dt.datetime})
data.index = ul.str_to_datetime (data.index.tolist())
######## NUMPY ARRAYS FORMAT
X_data = np.loadtxt(folder_dataFeatures + "Xdata.csv", delimiter=",")
price = np.loadtxt(folder_dataFeatures + "price.csv", delimiter=",")
price = price.reshape(Y_data.size,1) # TODO: Just to put it in the sahpe as it was before writing it to disk
dates = np.loadtxt(folder_dataFeatures + "dates.csv", delimiter=",")
## Generate the Y variable to estimate
lag = 20
Y_data = bMA.get_return(price, lag = lag)
Y_data = bMA.shift(Y_data, lag = -lag, cval = np.NaN)

if (model_OLS):
    # Makes use of the pandas structures
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
    # Peform analysis of variance on fitted linear model
    #anova_results = anova_lm(model)
    #print('\nANOVA results')
    #print(anova_results)

if (model_sklearn):
    # Makes use of the numpy structures
    ## Eliminate the Nans !! Even if they appear in just one dim
    mask_X = np.sum(np.isnan(X_data), axis = 1) == 0
    mask_Y = np.isnan(Y_data) == 0
    mask = mask_X & mask_Y[:,0]  # Mask without NaNs. This is done automatically in the OLS
    
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

if (lag_analysis):
    def get_Residuals_LinearModel(X_data, price, lag = 20):
        # This functions gets the residuals when we train a linear model with 
        # the X_data into predictiong the price return with a given lag.
    
        ## Prediction of the thingies
        Y_data = bMA.diff(price, lag = lag, cval = np.NaN)
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
        residuals.append(get_Residuals_LinearModel(X_data,price, lag = lag))

    gl.plot(lags, residuals)

############## Plotting of 3D regression ##############
plotting_3D = 1
if (plotting_3D):
    # We train 3 models 
#    Y_data = np.sign(Y_data)

    mask_X = np.sum(np.isnan(X_data), axis = 1) == 0
    mask_Y = np.isnan(Y_data) == 0
    mask = mask_X & mask_Y[:,0]  # Mask without NaNs. This is done automatically in the OLS
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    X_datafiltered = X_data[:,[0,1]]
    regr.fit(X_datafiltered[mask,:], Y_data[mask,:])
     #    coeffs = np.array([regr.intercept_, regr.coef_])[0]
    coeffs = np.append(regr.intercept_, regr.coef_)
    params = np.array(coeffs)
    
    gl.scatter_3D(X_data[:,0],X_data[:,1], Y_data,
                   labels = ["","",""],
                   legend = ["Pene"],
                   nf = 1)
                   
    grids = ul.get_grids(X_data)
    z = bMA.get_plane_Z(grids[0], grids[1],params)
    h = gl.plot_3D(grids[0],grids[1],z, nf = 0)
        