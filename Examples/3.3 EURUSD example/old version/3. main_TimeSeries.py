
# Change main directory to the main folder and import folders
import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")

os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Import functions independent of DataStructure
import utilities_lib as ul
import basicMathlib as bMl
import pandas as pd

import baseClassifiersLib as bCL
import toptal_utils as tut
from sklearn import linear_model

plt.close("all") # Close all previous Windows

"""
$$$$$$$$$$$$$$$$$$$$$$$ OPTIONS $$$$$$$$$$$$$$$$$$$$$$$$$
"""

folder_images = "../pics/Toptal/"
storage_folder = ".././storage/Toptal/"
file_name = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

load_data = 0
preprocessing_data = 0


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
    
"""
Create the target for the classifier:
"""

Target = bMl.diff(C).flatten()  # Continuous target Target[0] = NaN
Target_bin = np.zeros(Target.shape) # Binarized
Target_bin[np.where(Target >=0)] = 1

## Create Pandas Data Frame for the information of the ML problem

data_df = pd.DataFrame({'Time': days_keys, 'Target_clas': Target_bin,  'Target_reg': Target})  #  'Current_diff': Target} 
data_df.set_index('Time',inplace = True)

"""
#########################################################
CREATE WINDOWED VECTOR OF FEATURES !!
#########################################################
"""

Nlast_Close = 2  # The last Diff in Close and increase
for i in range(Nlast_Close):
#    data_df.shift()
    data_df["Diff_prevC_%i"%(i+1)] = bMl.shift(Target,lag = i+1)
    
    # We encode it as categorical !!! 
    data_df["Diff_prevC_bin_%i"%(i+1)] = bMl.shift(Target_bin,lag = i+1)
#    data_df["Diff_prevC_bin_%i"%(i+1)] = pd.Categorical(data_df["Diff_prevC_bin_%i"%(i+1)]).codes
    
Nlast_Range = 2
for i in range(Nlast_Range):
#    data_df.shift()
    data_df["Diff_prevRangeHL_%i"%(i+1)] = bMl.shift(H-L,lag = i+1)
    data_df["Diff_prevRangeCO_%i"%(i+1)] = bMl.shift(C-O,lag = i+1)

Nlast_Price = 1
for i in range(Nlast_Range):
#    data_df.shift()
    data_df["prevClose_%i"%(i+1)] = bMl.shift(C,lag = i+1)

#gl.scatter(data_df["Diff_prevC:%i"%(3)], data_df["Target_reg"])
#gl.scatter(data_df["Diff_prevRangeHL:%i"%(1)], data_df["Diff_prevRangeCO:%i"%(1)])
#gl.scatter(data_df["prevClose:%i"%(1)], data_df["Diff_prevRangeHL:%i"%(1)])

"""
#########################################################
CREATE WINDOWED VECTOR OF FEATURES !!
#########################################################
"""
Driving_volume = []
# We would like to have an indicator that tell us how many contracts of the volume have been used
# to move the price up, and how many to move it down. This is hard to know from daily data but we can get a better representation from 15M

def get_DV(TD):
    DV = np.sum((TD["Close"]-TD["Open"])*TD["Volume"])/(np.sum(TD["Volume"])*(np.max(TD["High"] -np.min(TD["Low"]))))
    return DV

for day in days_keys:
    Driving_volume.append(get_DV(timeData.TD.ix[day_dict[day]]))


Nlast_DV = 3
for i in range(Nlast_Range):
#    data_df.shift()
    data_df["DV_%i"%(i+1)] = bMl.shift(Driving_volume,lag = i+1)
    
"""
######################################################
############# Get the final data structures for the algorihtms 
#####################################################
"""

"""
######################################################
############# Calculate correlations !
#####################################################
"""

PropTrain = 0.7

# Remove the samples that did not have enough previous data !!!
data_df.dropna(inplace = True)
Nsa, Nd = data_df.shape

X = np.array(data_df.loc[:, np.logical_and(data_df.columns != 'Target_clas' , data_df.columns != 'Target_reg')])
Y = np.array(data_df.loc[:,'Target_clas'])


Xtrain = X[:int(PropTrain * Nsa),:]
Xtest = X[int(PropTrain * Nsa):,:]
Ytrain = Y[:int(PropTrain * Nsa)].flatten()
Ytest = Y[int(PropTrain * Nsa):].flatten()

Ytrain_reg = data_df["Target_reg"][:int(PropTrain * Nsa)]
Ytest_reg = data_df["Target_reg"][int(PropTrain * Nsa):]

dates_train = data_df.index[:int(PropTrain * Nsa)]
dates_test = data_df.index[int(PropTrain * Nsa):]

model_sklearn = 1
if (model_sklearn):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(Xtrain, Ytrain_reg)
    
    #    coeffs = np.array([regr.intercept_, regr.coef_])[0]
    coeffs = np.append(regr.intercept_, regr.coef_)
    params = np.array(coeffs)

    residual = regr.residues_
    print("sklearn model Parameters")
    print(params)
    print("Residual")
    print (residual)



import graph_tsa as grtsa
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
#
#returns = timeData.get_timeSeriesReturn()
#grtsa.plot_acf_pacf(Ytrain)
#grtsa.plot_decomposition(C)
    # Fit the model

input_columns = data_df.columns[np.logical_and(data_df.columns != 'Target_clas' , data_df.columns != 'Target_reg')]
## For categorical values we have to put C(name variable)
model = ols("Target_reg ~ " + " + ".join(input_columns), data_df[:][:int(PropTrain * Nsa)]).fit()
params = model._results.params
# Print the summary
print(model.summary())
print("OLS model Parameters")
print(params)


## Peform analysis of variance on fitted linear model
model = ols("Target_reg ~  Diff_prevC_bin_1 ", data_df).fit()
anova_results = anova_lm(model)
print('\nANOVA results')
print(anova_results)


"""
#####################
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

"""

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

