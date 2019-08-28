"""
This file implements several classifications systems over the selected features
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
import t_utils as tut
import baseClassifiersLib as bCL
import basicMathlib as bMl
import indicators_lib as indl
import pickle_lib as pkl 
import utilities_lib as ul
plt.close("all") # Close all previous Windows

# %% 
"""
################### EXECUTING OPTIONS ###################
"""

folder_images = "../pics/EURUSD/"
storage_folder = ".././storage/EURUSD/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

load_data = 1
preprocessing_data = 1
extract_features = 1
try_classifiers = 1
save_model_to_disk = 1

plot_performance_all = 1
plot_results = 1

# Using the library of function built in using the dataFrames in pandas
typeChart = "Bar"  # Line, Bar, CandleStick
tranformIntraday = 1

symbols = ["EURUSD"]
periods = [15]  # 1440 15

######## SELECT DATE LIMITS ###########
## We set one or other as a function of the timeSpan

sdate_str = "01-01-2010"
edate_str = "31-12-2016"

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
    """
    Final Subselection of Features, normalization and data splitting
    """
    data_df = data_df[["Target_clas","Target_reg","week_1","Target_1","Target_2",
                       "Target_3","RSI_1","MACD_1","ACCDIST_1"]]
    # Remove the samples that did not have enough previous data !!!
    data_df.dropna(inplace = True)
    PropTrain = 0.7
    Nsa, Nd = data_df.shape
    last_tr_sa = int(PropTrain * Nsa)
    
    data_df_train = data_df[:][:last_tr_sa]
    data_df_test = data_df[:][last_tr_sa:]
    
    input_features_names = data_df.columns[np.logical_and(data_df.columns != 'Target_clas' , 
                                                          data_df.columns != 'Target_reg')]
    
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
    scaler_X = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler_X.transform(Xtrain)            
    Xtest = scaler_X.transform(Xtest)       
    data_df_train[input_features_names] = scaler_X.transform(data_df_train[input_features_names] )            
    data_df_test[input_features_names] = scaler_X.transform(data_df_test[input_features_names])     
    
    # Continuous target
    scaler_Y = preprocessing.StandardScaler().fit(Ytrain_reg)
    Ytrain_reg = scaler_Y.transform(Ytrain_reg).flatten()         
    Ytest_reg = scaler_Y.transform(Ytest_reg).flatten()   
    
    data_df_train["Target_reg"] = Ytrain_reg        
    data_df_test["Target_reg"] = Ytest_reg

    
# %%
"""
#####################################################
################# Classifiers  #################
#####################################################
"""

### Try a lot of commonly used classifiers !! 
if (try_classifiers):
    lr = bCL.get_LogReg(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
    lda = bCL.get_LDA(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
    qda = bCL.get_QDA(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
    gnb =  bCL.get_GNB(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
    gknn =  bCL.get_KNN(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
        
    
    ## Treee !
    gtree = bCL.get_TreeCl(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
    rf = bCL.get_RF(Xtrain, Ytrain,gtree, Xtest, Ytest, verbose = 1)
    ert = bCL.get_ERT(Xtrain, Ytrain,gtree, Xtest, Ytest, verbose = 1)
    
    ## SVMs
    gsvmr = bCL.get_LSVM(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
    sbmrf =  bCL.get_SVM_rf(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)
#    gsvmp = bCL.get_SVM_poly(Xtrain, Ytrain, Xtest, Ytest, verbose = 1)

    ## Create dict of classifiers
    cl_d = dict()
    cl_d["LR"] = lr; cl_d["LDA"] = lda;cl_d["QDA"] = qda; cl_d["GNB"] = gnb;
    cl_d["KNN"] = gknn; cl_d["Tree"] = gtree; cl_d["RF"] = rf; cl_d["ERF"] = ert;
    cl_d["LSVM"] = gsvmr; cl_d["RFSVM"] = sbmrf; 
#    cl_d["PolySVM"] = gsvmp;
    
if (save_model_to_disk):
    # We save the last model to disk using pickle ! 
    # Lazy way but since non Deep Learning models do not require much 
    # memory in general, it is justified.
    folder_model = "../models/"
    key_classifier = "LSVM"  # QDA  # GNB RF
    
    ul.create_folder_if_needed(folder_model)
    classifier = cl_d[key_classifier]
    pkl.store_pickle(folder_model + key_classifier +".pkl", [classifier])
    
    pkl.store_pickle(folder_model + "scaler_X" +".pkl", [scaler_X])
    pkl.store_pickle(folder_model + "scaler_Y" +".pkl", [scaler_Y])
     
"""
PLOT THE RESULTS
"""

if (plot_performance_all):
    
    def get_class_rate(Y, Ypred):
        return np.mean(np.equal(Y,Ypred))
    
    def get_CE (Y, Ypred):
         tol = 1e-5
         return -np.mean(Y*np.log(Ypred +tol) + (1-Y)*np.log(1-Ypred + tol))
         
    train_acc = []
    test_acc = []
    
    train_CE = []
    test_CE = []
    
    classifiers_keys = cl_d.keys()
    Nclassifiers = len(classifiers_keys)
    for key in classifiers_keys:
        classifier = cl_d[key]
        train_acc.append(get_class_rate(Ytrain, classifier.predict(Xtrain)))
        test_acc.append(get_class_rate(Ytest, classifier.predict(Xtest)))
        
        train_CE.append(get_CE(Ytrain, classifier.predict_proba(Xtrain)[:,1]))
        test_CE.append(get_CE(Ytest, classifier.predict_proba(Xtest)[:,1]))
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    train_CE = np.array(train_CE)
    test_CE = np.array(test_CE)
    
    gl.init_figure()
    ax1 = plt.subplot(2,1,1)
    plt.bar(np.arange(Nclassifiers)+0.2,1-train_acc,width=0.2,color='c',align='center', label = "train")
    plt.bar(np.arange(Nclassifiers)+0.4,1-test_acc,width=0.2,color='r',align='center', label = "test")
    plt.xticks(np.arange(Nclassifiers)+0.3,classifiers_keys)
    plt.title('Classifiers Performance')
    plt.ylabel('Error rate')
    plt.grid()
    
    ax2 = plt.subplot(2,1,2)
    plt.bar(np.arange(Nclassifiers)+0.2,train_CE,width=0.2,color='c',align='center',label = "train")
    plt.bar(np.arange(Nclassifiers)+0.4,test_CE,width=0.2,color='r',align='center', label = "test")
    plt.xticks(np.arange(Nclassifiers)+0.3,classifiers_keys)
#    plt.title('Classifiers CE')
    plt.ylabel('CE')
    plt.grid()
    plt.show()
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.2)
    
    gl.set_fontSizes(ax = [ax1,ax2,], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 18, yticks = 12)
    
    gl.savefig(folder_images +'Classifiers_performance.png', 
           dpi = 100, sizeInches = [3*8, 3*2])
    
if (plot_results):
    
    key_classifier = "QDA"  # QDA  # GNB RF
    classifier = cl_d[key_classifier]
    # Compute how well we have done in each sample using cross entropy
    Ypredict_test_proba = classifier.predict_proba(Xtest)[:,1] # probability of 1 
    Ypredict_train_proba = classifier.predict_proba(Xtrain)[:,1]
    Ypredict_test = classifier.predict(Xtest)
    Ypredict_train = classifier.predict(Xtrain)
    
    test_cross_entropy = Ytest*np.log(Ypredict_test_proba) + (1-Ytest)*np.log(1-Ypredict_test_proba)
    train_cross_entropy = Ytrain*np.log(Ypredict_train_proba) + (1-Ytrain)*np.log(1-Ypredict_train_proba)
    
    test_cross_entropy = -test_cross_entropy
    train_cross_entropy = -train_cross_entropy
    
    ##############################################################
    alpha_stem = 0.5
    marker_stem = [".",1,None]
    train_color = "b" 
    test_color = "r"
    
    gl.init_figure();
    ax1 = gl.subplot2grid((4,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((4,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = gl.subplot2grid((4,1), (2,0), rowspan=1, colspan=1, sharex = ax1)
    ax4 = gl.subplot2grid((4,1), (3,0), rowspan=1, colspan=1, sharex = ax1)
    
    ## Ax1 = Close price at the end of the sessions
    gl.plot(days_keys, C, ax = ax1, labels = ["Results " + key_classifier,"","Close Rate"],
            AxesStyle = "Normal - No xaxis", legend = ["Close Rate"])
    
    ## Ax2 = 1 if the stock has gone up, zero if it has gone down
    gl.stem(dates_train, Ytrain_reg, ax = ax2, labels = ["","","Target_reg"], bottom = 0.0, AxesStyle = "Normal - No xaxis", alpha = alpha_stem,
            marker = marker_stem, color = train_color, legend = ["tr"])
    gl.stem(dates_test, Ytest_reg, ax = ax2, labels = ["","","Target_reg"],bottom = 0.0,AxesStyle = "Normal - No xaxis", color = test_color ,
            alpha = alpha_stem, marker = marker_stem, legend = ["tst"])
    
    # Ax3 = The estimates probability for train and test
    gl.stem(dates_train,Ypredict_train_proba ,bottom = 0.5, ax = ax3, labels = ["","",r"$\hat{Y}$"],AxesStyle = "Normal - No xaxis", 
            alpha =alpha_stem, marker = marker_stem,  color = train_color, legend = ["tr"])
    gl.stem(dates_test, Ypredict_test_proba,bottom = 0.5, ax = ax3, labels = ["","",r"$\hat{Y}$"],color = test_color,ylim = [0.0,1.0],alpha = alpha_stem,
            AxesStyle = "Normal - No xaxis", marker = marker_stem, legend = ["tst"]) # ylim = [0.0,1.0]
    
    ## Ax4 = Cross Entropy of the samples for train and test
    gl.stem(dates_train,train_cross_entropy ,bottom = 0.0, ax = ax4, labels = ["","","CE"],AxesStyle = "Normal", 
            alpha = alpha_stem, marker = marker_stem,  color = train_color, legend = ["tr"])
    gl.stem(dates_test, test_cross_entropy,bottom = 0.0, ax = ax4, color = test_color,AxesStyle = "Normal", 
            alpha = alpha_stem,marker = marker_stem, legend = ["tst"])
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.05, hspace=0.05)
    
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 12, yticks = 12)
    
    image_name = "Results_" + key_classifier
    gl.savefig(folder_images + image_name, 
           dpi = 100, sizeInches = [20, 7])
    
