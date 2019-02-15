"""
This file explores different Feature Selection algorithms over the extracted
features 
"""
# %% 
# Load all the directories needed for the code to be executed both from
# a console and Spyder, and from the main directory and the local one.
# Ideally this code should be executed using Spyder with working directory

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Specific utilities
import indicators_lib as indl
import t_utils as tut
import basicMathlib as bMl

plt.close("all") # Close all previous Windows

# %% 
"""
################### EXECUTING OPTIONS ###################
"""

folder_images = "../pics/EURUSD/"
storage_folder = ".././storage/EURUSD/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

# flags to load and preprocessed data (for rime optimization)
load_data = 1
preprocessing_data = 1
extract_features = 1

random_forest_selection = 1
SVM_selection = 1
Linear_Model_selection = 1
plot_correlation = 1

# Symbol information
symbols = ["EURUSD"]
periods = [15]  # 1440 15

######## SELECT DATE LIMITS ###########
## We set one or other as a function of the timeSpan
sdate_str = "01-01-2010"
edate_str = "31-12-2016"
#edate_str = "31-1-2011"
sdate = dt.datetime.strptime(sdate_str, "%d-%m-%Y")
edate = dt.datetime.strptime(edate_str, "%d-%m-%Y")

# %% 
if (load_data):
    ######## CREATE THE OBJECT AND LOAD THE DATA ##########
    # Tell which company and which period we want
    timeData = CTD.CTimeData(symbols[0],periods[0])
    timeData.set_csv(storage_folder,file_name)  # Load the data into the model

if (preprocessing_data):
    timeData = tut.preprocess_data(timeData, sdate,edate)
    ## Get the valid trading days sorted
    days_keys, day_dict = timeData.get_indexDictByDay()
    Ndays = len(days_keys)

    timeData_daily = tut.get_daily_timedata(timeData, symbols[0])
    H,L,O,C,V = np.array(timeData_daily.TD[["High","Low","Open","Close","Volume"]][:]).T
    

if (extract_features):
    """
    Create the target for the regressing and classifier systems
    """
    Target = bMl.diff(C).flatten()  # Continuous target .Target[0] = NaN
    Target_bin = np.zeros(Target.shape) # Binarized target
    Target_bin[np.where(Target >=0)] = 1
    data_df = None
    ## Create Pandas Data Frame for the information of the ML problem
    data_df = pd.DataFrame({'Time': days_keys, 'Target_clas': Target_bin,  'Target_reg': Target})
    data_df.set_index('Time',inplace = True)
    
    ## Every feature that is computed for each day has to be lagged before introducing 
    ## it into the dataframe so that the input features do not contain information
    ## from their target day
    
    ################  Time specific variables ##################
    # Used for algorithms to learn seasonal patterns and proximity between samples.
    # We could just give the date but if we separate in different components it is 
    # easier for algorithms to learn from it
    day_of_week = np.array(data_df.index.dayofweek)
    week_of_year = np.array(data_df.index.weekofyear)
    year = np.array(data_df.index.year)
    
    ## Add the lagged value to the database
    Nlag_time_information = 1
    tut.add_lagged_values(data_df,day_of_week,"day",Nlag_time_information)
    tut.add_lagged_values(data_df,week_of_year,"week",Nlag_time_information)
    tut.add_lagged_values(data_df,year,"year",Nlag_time_information)
    
    ################  OCHL variables ##################
    # Variables trivially obtained from daily OHCL
    Target = Target # Increase in Close price
    Range_HL = H-L # measure of volatility
    Daily_gap =  O - bMl.shift(C,lag = 1).flatten() # measure of price movement
    
    ## Add the lagged value to the database
    Nlag_OCHL_information = 3
    tut.add_lagged_values(data_df,Target,"Target",Nlag_OCHL_information)
    tut.add_lagged_values(data_df,Range_HL,"Range_HL",Nlag_OCHL_information)
    tut.add_lagged_values(data_df,Daily_gap,"Daily_gap",Nlag_OCHL_information)
    
    
    ################## Daily Trading Indicators ####################
    # Hulls_average !! ACDC, Volatility, ATR, Short 
    nHMA = 20
    ## Hulls Average, reactive but smoothing MA
    HMA  = indl.get_HMA(timeData_daily.get_timeSeries(["Close"]), nHMA)  
    
    ## Volatility
    nAHLR = 20; nBB = 20; nATR = 20; nCha = 20;
    AHLR = timeData_daily.AHLR(n = nAHLR)
    ATR = timeData_daily.ATR(n = nATR)
    EMA_Range, Cha = timeData_daily.Chaikin_vol(n = nCha)
    BB = timeData_daily.BBANDS(seriesNames = ["Close"], n = nBB)
    BB = BB[:,0] - BB[:,1] 
    
    # Oscillators
    n , SK, SD = 14, 6,6
    L = 14
    L1 , L2, L3 = 14, 9,12
    
    STO = timeData_daily.STO(n = n, SK = SK, SD = SD)
    RS, RSI = timeData_daily.RSI(n = L)
    EX1,EX2,EX3,TRIX = timeData_daily.TRIX(L1 , L2, L3)
    MACD, MACDsign, MACDdiff = timeData_daily.MACD().T
        
    # Volume related
    nAD = 5;
    ACCDIST = timeData_daily.ACCDIST(n = nAD)
    DV = timeData_daily.ACCDIST(n = nAD)
    
    ## Add variables to the data_frame
    Nlag_trading_info = 1
    tut.add_lagged_values(data_df,HMA,"HMA",Nlag_trading_info)
    tut.add_lagged_values(data_df,RSI,"RSI",Nlag_trading_info)
    tut.add_lagged_values(data_df,MACD,"MACD",Nlag_trading_info)
    tut.add_lagged_values(data_df,ACCDIST,"ACCDIST",Nlag_trading_info)
    
# %% 
#######################################
### Divide in train and test  #########
#######################################

# Remove the samples that did not have enough previous data !!!
data_df.dropna(inplace = True)

PropTrain = 0.7
Nsa, Nd = data_df.shape
last_tr_sa = int(PropTrain * Nsa)

data_df_train = data_df[:][:last_tr_sa]
data_df_test = data_df[:][last_tr_sa:]

input_features_names = data_df.columns[np.logical_and(data_df.columns != 'Target_clas' , data_df.columns != 'Target_reg')]


X = np.array(data_df.loc[:,input_features_names])
Y = np.array(data_df.loc[:,'Target_clas'])

Xtrain = X[:last_tr_sa,:]
Xtest = X[last_tr_sa:,:]
Ytrain = Y[:last_tr_sa].flatten()
Ytest = Y[last_tr_sa:].flatten()

Ytrain_reg = np.array(data_df["Target_reg"][:int(PropTrain * Nsa)]).reshape(-1,1)
Ytest_reg = np.array(data_df["Target_reg"][int(PropTrain * Nsa):]).reshape(-1,1)

dates_train = data_df.index[:int(PropTrain * Nsa)]
dates_test = data_df.index[int(PropTrain * Nsa):]

## Normalize variables !!! 
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)       
data_df_train[input_features_names] = scaler.transform(data_df_train[input_features_names] )            
data_df_test[input_features_names] = scaler.transform(data_df_test[input_features_names])     

# Continuous target
scaler = preprocessing.StandardScaler().fit(Ytrain_reg)
Ytrain_reg = scaler.transform(Ytrain_reg).flatten()         
Ytest_reg = scaler.transform(Ytest_reg).flatten()   

data_df_train["Target_reg"] = Ytrain_reg        
data_df_test["Target_reg"] = Ytest_reg
   # %% 
"""
#####################################################
################# Feature Seleciton #################
#####################################################
"""

if (random_forest_selection):
    # Build a forest and compute the feature importances
    from sklearn.ensemble import ExtraTreesClassifier
    forest = ExtraTreesClassifier(n_estimators=1000,
                                  random_state=5)
    
    forest.fit(Xtrain, Ytrain)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    
    indices = np.argsort(importances)[::-1]
    names = input_features_names[indices]
    
    ##################################################################
    ###########  Plot the feature importances of the forest ###########
    gl.init_figure()
    plt.title("Feature importances Random Forest")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    
    plt.xticks(range(X.shape[1]), names, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()
    gl.savefig(folder_images +'Feature_importance_RF.png', 
           dpi = 100, sizeInches = [2*8, 2*2])
# %% 
if (SVM_selection):
    from sklearn.svm import LinearSVC
    
    ## We train by cross validation a Linear SVM and the importance of 
    ## each feature is given by the magnitude of its weight
    
    lsvc = LinearSVC(C=0.02, penalty="l1", dual=False).fit(Xtrain, Ytrain)
    coef  = lsvc.coef_
    importances = coef.flatten()

    indices = np.argsort(importances)[::-1]
    names = input_features_names[indices]

    ##################################################################
    ###########  Plot the feature importances of the SVC ###########
    
    gl.init_figure()
    plt.title("Feature importances of SVC")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", align="center")
    
    plt.xticks(range(X.shape[1]), names, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()
    gl.savefig(folder_images +'Feauture_importance_SVC.png', 
           dpi = 100, sizeInches = [2*8, 2*2])
# %% 
if (Linear_Model_selection):

    from statsmodels.formula.api import ols

    ## For categorical values we have to put C(name variable)
    model = ols("Target_reg ~ " + " + ".join(input_features_names), data_df_train[:][:last_tr_sa]).fit()
    params = model._results.params
    
    importances = params[1:]  # Remove the intrcept
    pvalues = model._results.pvalues[1:]
    std = np.diag(model.cov_params())[1:]
    
    indices = np.argsort(importances)[::-1]
    names = input_features_names [indices]
    # Print the summary
    print(model.summary())
    print("OLS model Parameters")

    ##################################################################
    ###########  Plot the feature importances of the SVC ###########
    
    gl.init_figure()
    plt.title("Feature importances Linear Model")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", align="center", yerr=std[indices])
    
    plt.xticks(range(X.shape[1]), names, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    
    plt.stem(range(X.shape[1]), pvalues[indices], label = "p-value")
    plt.show()

    gl.savefig(folder_images +'Feauture_importance_LM.png', 
           dpi = 100, sizeInches = [2*8, 2*2])
        
    ############## Plot the correlation matrix #######################
    if (plot_correlation):
        Cov_tr = np.corrcoef(data_df_train.T)
        from matplotlib import cm as cm
    #    
        gl.init_figure();
        ax1 = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=4)
    
        cmap = cm.get_cmap('jet', 30)
        cax = ax1.imshow(Cov_tr, interpolation="nearest", cmap=cmap)
        plt.xticks(range(data_df_train.shape[1]), data_df_train.columns, rotation='vertical')
        plt.yticks(range(data_df_train.shape[1]), data_df_train.columns, rotation='horizontal')
        plt.colorbar(cax)
    #        ax1.set_xticks(data_df_train.columns) # , rotation='vertical'
    #    ax1.grid(True)
        plt.title('Correlation matrix of variables')
    #    labels=[str(x) for x in range(Nshow )]
    #    ax1.set_xticklabels(labels,fontsize=20)
    #    ax1.set_yticklabels(labels,fontsize=20)
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        plt.show()
        gl.savefig(folder_images +'Corr.png', 
               dpi = 100, sizeInches = [2*8, 2*2])


