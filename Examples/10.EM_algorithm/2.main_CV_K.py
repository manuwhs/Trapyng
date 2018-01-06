"""
This code will:
    - Perform the CV of the number of clusters for a given set of data.
"""

import os
os.chdir("../../")
import import_folders
# Classical Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy as copy
# Data Structures Data
import CPortfolio as CPfl

# Own graphical library
from graph_lib import gl 
# Import functions independent of DataStructure
import utilities_lib as ul
# For statistics. Requires statsmodels 5.0 or more
# Analysis of Variance (ANOVA) on linear models
import basicMathlib as bMA

import DDBB_lib as DBl

import CEM as CEM 
import CDistribution as Cdist

import specific_plotting_func as spf
from sklearn import cross_validation
gl.close("all")

##############################################
########## FLAGS ############################
trading_graph = 1

perform_EM = 1
perform_HMM_after_EM = 1
final_clusters_graph = 1
responsibility_graph = 1
##########################################################################
################# DATA OBTAINING ######################################
##########################################################################
######## SELECT SOURCE ########
dataSource =  "Google"  # Hanseatic  FxPro GCI Yahoo
[storage_folder, info_folder, 
 updates_folder] = ul.get_foldersData(source = dataSource)
folder_images = "../pics/Trapying/EM_trading/"
ul.create_folder_if_needed(folder_images)

######## SELECT SYMBOLS AND PERIODS ########
symbols = ["XAUUSD","Mad.ITX", "EURUSD"]
symbols = ["Alcoa_Inc"]
symbols = ["GooG", "Alcoa_Inc"]
periods = [5]

# Create porfolio and load data
cmplist = DBl.read_NASDAQ_companies(whole_path = "../storage/Google/companylist.csv")
cmplist.sort_values(by = ['MarketCap'], ascending=[0],inplace = True)
symbolIDs = cmplist.iloc[0:30]["Symbol"].tolist()

symbol_ID_indx1 = 0
symbol_ID_indx2 = 1

for period in periods:
    Cartera = CPfl.Portfolio("BEST_PF", symbolIDs, [period]) 
    Cartera.set_csv(storage_folder)
    
sdate = dt.datetime.strptime("15-8-2017", "%d-%m-%Y")
edate = dt.datetime.strptime("1-9-2017", "%d-%m-%Y")
#edate = dt.datetime.now()

Cartera.set_interval(sdate, edate)

opentime, closetime = Cartera.get_timeData(symbolIDs[symbol_ID_indx1],periods[0]).guess_openMarketTime()
dataTransform = ["intraday", opentime, closetime]


######### FILL THE DATA #####################
## TODO: How are we filling the data ? 

fill_data_f = 0
if (fill_data_f):
    Cartera.get_timeData(symbolIDs[symbol_ID_indx1],periods[0]).fill_data()
    Cartera.get_timeData(symbolIDs[symbol_ID_indx2],periods[0]).fill_data()

########## Obtain the data ###############
ret1 = Cartera.get_timeData(symbolIDs[symbol_ID_indx1],periods[0]).get_timeSeriesReturn()*100
ret2 = Cartera.get_timeData(symbolIDs[symbol_ID_indx2],periods[0]).get_timeSeriesReturn()*100

dates = Cartera.get_timeData(symbolIDs[symbol_ID_indx1],periods[0]).get_dates()
dates2 = Cartera.get_timeData(symbolIDs[symbol_ID_indx2],periods[0]).get_dates()


##########################################################################
################# PREPROCESS DATA ######################################
##########################################################################

## Set GAP return as NAN
remove_gap_return = 1
if (remove_gap_return):
    """ We usually would like to remove the return of gaps if we are dealing
        with intraday data since they are ouliers for this distribution,
        they belong to a distribution with more time
    """
    # If we had all the data properly this would do.
    if(0):
        gap_ret = np.where(dates.time == opentime)[0]
    else:
    # If we do not, we need to find the first sample of each day.
        days_keys, day_dict = Cartera.get_timeData(symbolIDs[symbol_ID_indx1],periods[0]).get_indexDictByDay()
        gap_ret = [day_dict[k][0] for k in days_keys]
#    print ("Indexes: ",day_dict[days_keys[0]])
    
    ret1[gap_ret,:] = np.NaN
    ret2[gap_ret,:] = np.NaN


    ## Remove the NaNs
    NonNan_index =  np.logical_not(np.isnan(ret1) | np.isnan(ret2))
    ret1 = ret1[NonNan_index[:,0],:]
    ret2 = ret2[NonNan_index[:,0],:]
    dates = dates[NonNan_index[:,0]]

#### Apply some shifting if we want crosscorrelations though time
lag_sigmals = 0
if(lag_sigmals != 0 ):
    ret2 = ret2[lag_sigmals:,:]
    lag_sigmals = ret1[:-lag_sigmals,:]
    
## Final data

data = np.concatenate((ret1,ret2),axis = 1)
mean = np.mean(data, axis = 0)
corr = bMA.get_corrMatrix(data)
cov = bMA.get_covMatrix(data)


data = data - np.mean(data, axis = 0)

""" 
Final preprocessing of the data.
We can either put it all in a single chain, or put the chains by days
"""

if (0):
    Xdata = [data]
else:
    TD_new_filtered_dates = pd.DataFrame(dates)
    TD_new_filtered_dates.index = dates
    days_keys, day_dict = Cartera.get_timeData(symbolIDs[symbol_ID_indx1],periods[0]).get_indexDictByDay(TD = TD_new_filtered_dates)
    Xdata = [data[day_dict[k],:] for k in days_keys]

    Nchains = len(Xdata)
    
if (perform_EM):
    ######## Create the Distribution object ##############################################
    
    #### 1st Gaussian Distribution
    Gaussian_d = Cdist.CDistribution(name = "Gaussian");
    Gaussian_d.set_distribution("Gaussian")
    Gaussian_d.parameters["mu_variance"]  = 0.2
    Gaussian_d.parameters["Sigma_min_init"] = 0.2
    Gaussian_d.parameters["Sigma_max_init"] = 0.3
    Gaussian_d.parameters["Sigma_min_singularity"] = 0.1
    Gaussian_d.parameters["Sigma_min_pdf"] = 0.1
    Gaussian_d.parameters["Sigma"] = "full"
    
    #### 1st Gaussian Distribution
    Gaussian_d2 = Cdist.CDistribution(name = "Gaussian2");
    Gaussian_d2.set_distribution("Gaussian")
    Gaussian_d2.parameters["mu_variance"]  = 0.2
    Gaussian_d2.parameters["Sigma_min_init"] = 0.2
    Gaussian_d2.parameters["Sigma_max_init"] = 0.3
    Gaussian_d2.parameters["Sigma_min_singularity"] = 0.1
    Gaussian_d2.parameters["Sigma_min_pdf"] = 0.1
    Gaussian_d2.parameters["Sigma"] = "diagonal"
    
    #### 2nd: Watson distribution
    Watson_d = Cdist.CDistribution(name = "Watson");
    Watson_d.set_distribution("Watson")
    Watson_d.parameters["Num_Newton_iterations"] = 5
    Watson_d.parameters["Allow_negative_kappa"] = "no"
    Watson_d.parameters["Kappa_max_init"] = 100
    Watson_d.parameters["Kappa_max_singularity"] =1000
    Watson_d.parameters["Kappa_max_pdf"] = 1000
    Watson_d.use_changeOfClusters = None
    #### 3rd von MisesFisher distribution ######
    vonMisesFisher_d = Cdist.CDistribution(name = "vonMisesFisher");
    vonMisesFisher_d.set_distribution("vonMisesFisher")
    
    vonMisesFisher_d.parameters["Num_Newton_iterations"] = 2
    vonMisesFisher_d.parameters["Kappa_max_init"] = 20
    vonMisesFisher_d.parameters["Kappa_max_singularity"] =1000
    vonMisesFisher_d.parameters["Kappa_max_pdf"] = 1000

    ############# SET TRAINNG PARAMETERS ##################################
    
    Ninit = 10
    delta_ll = 0.02
    T = 50
    verbose = 1;
    clusters_relation = "independent"   # MarkovChain1  independent
    time_profiling = None
    
    ### Number of clusters to crossvalidate
    Klusters =  [1,2,3,4,5]  # range(1,8) # 3,4,5,6,10,10,12,15
    NCVs = 1    # Number of times we do the crossvalidation
    Nfolds =  Nchains  # Number of folds of each crossvalidation
    

    ### Data structures to save the results
    logl_tr_CVs = []  # List of the training LL of all CVs   [nCVs]
    logl_val_CVs = [] # List of the validation LL of all CVs   [nCVs]
    
    
    ############################# Begin CV  ####################################
    fake_labels = np.ones(Nchains).flatten() # Emulation of all samples have the same class
    
    for nCV_i in range(NCVs):   # For every CV that we will perform
        logl_tr_CV_i = []   # List the LL values for a single CV, for all clusters and Nfolds. [NKluters] = [LL_fold1, LL_fold2,....]]
        logl_val_CV_i = []  # List the LL values for a single CV, for all clusters and Nfolds. [NKluters] = [LL_fold1, LL_fold2,....]]
        # We create the splitting of indexes
        stkfold = cross_validation.StratifiedKFold(fake_labels, n_folds = Nfolds, shuffle= True)
        # We are going to use the same partition for all parameters to crossvalidate.
        # In this case the number of clusters
        for K in Klusters:  
            
            # Create the distributino manager with those clusters
            myDManager = Cdist.CDistributionManager()
            K_G = 0   # Number of clusters for the Gaussian Distribution
            K_G2 = K # Number of clusters for the Gaussian Distribution
            K_W = 0
            K_vMF = 0
            if (K_G > 0):
                myDManager.add_distribution(Gaussian_d, Kd_list = range(0,K_G))
            if(K_W > 0):
                myDManager.add_distribution(Watson_d, Kd_list = range(K_G,K_W+ K_G))
            if(K_vMF > 0):
                myDManager.add_distribution(vonMisesFisher_d, Kd_list = range(K_W+ K_G,K_W+ K_G+ K_vMF))
            if (K_G2 > 0):
                myDManager.add_distribution(Gaussian_d2, Kd_list = range(K_W+ K_G+ K_vMF,K_W+ K_G+ K_vMF+K_G2))
                
            print "CrossValidating Number of Clusters %i" % K
            ll_tr_params_folds = []
            ll_val_params_folds = []
            ifold = 1
            for train_index, val_index in stkfold:  # For each partition, using the set of parameters to CV
                print "Starting fold out of %i/%i"%(ifold,Nfolds)
                ifold = ifold + 1
                Xdata_train = np.concatenate( [Xdata[indx] for indx in train_index], axis = 0)
                Xdata_val = np.concatenate( [Xdata[indx] for indx in val_index], axis = 0)
            
#                print Xdata_train.shape
#                print Xdata_val.shape
                # Create the EM object with the hyperparameters
                myEM = CEM.CEM( distribution = myDManager, clusters_relation = clusters_relation, 
                T = T, Ninit = Ninit,  delta_ll = delta_ll, 
                verbose = verbose, time_profiling = time_profiling)
                ## Perform EM with the hyperparameters already set, we just fit the data and position init
                theta_init = None; model_theta_init = None
                ############# PERFORM THE EM #############
                logl,theta_list,mode_theta_list = myEM.fit(Xdata_train, model_theta_init = model_theta_init, theta_init = theta_init) 
        
                ## Compute the likelihoods for train and test divided by the number of samples !! 
                ## We are computing the normalized Likelihood !! Likelihood per sample !!
                new_ll = myEM.get_loglikelihood(Xdata_train,myDManager, theta_list[-1],mode_theta_list[-1])/train_index.size
                ll_tr_params_folds.append(copy.deepcopy(new_ll))
                new_ll = myEM.get_loglikelihood(Xdata_val,myDManager, theta_list[-1],mode_theta_list[-1])/val_index.size
                ll_val_params_folds.append(copy.deepcopy(new_ll))
            
            logl_tr_CV_i.append(copy.deepcopy(ll_tr_params_folds))
            logl_val_CV_i.append(copy.deepcopy(ll_val_params_folds))
    
        logl_tr_CVs.append(logl_tr_CV_i)
        logl_val_CVs.append(logl_val_CV_i)

    # Create alterego variables for using in emergency later
    logl_tr_CVs_EM_save_for_later = copy.deepcopy(logl_tr_CVs)
    logl_val_CVs_EM_save_for_later = copy.deepcopy(logl_val_CVs)
    
    ################################################################################################################
    ################################### Put the data in proper format for plotting #####################################################
    ################################################################################################################
    ### Reschedule the data. We are going to for a single CV:
    # Compute the mean LL of the CVs for train and validation

    logl_tr_CVs = logl_tr_CVs_EM_save_for_later
    logl_val_CVs = logl_val_CVs_EM_save_for_later
    
    for i in range(len(logl_tr_CVs)):
        mean_tr_ll =  []
        mean_val_ll = []
        std_tr_ll = []
        std_val_ll = []
        for k_i in range(len(Klusters)):
            mean_tr_ll.append(np.mean(logl_tr_CVs[i][k_i]))
            mean_val_ll.append(np.mean(logl_val_CVs[i][k_i]))
            std_tr_ll.append(np.std(logl_tr_CVs[i][k_i]))
            std_val_ll.append(np.std(logl_val_CVs[i][k_i]))
            # For each CVs
    
        mean_tr_ll = np.array(mean_tr_ll)
        mean_val_ll= np.array(mean_val_ll)
        std_tr_ll= np.array(std_tr_ll)
        std_val_ll= np.array(std_val_ll)
        
    ################################################################################################################
    ################################### PLOTTING THE RESULTS #####################################################
    ################################################################################################################
    
    gl.init_figure()
    gl.plot(Klusters,mean_tr_ll, legend = ["Mean Train LL (EM)"], 
            labels = ["Validation of Number of clusters for a %i-CV EM."%Nfolds,"Number of clusters (K)","Average LL of a sample"], 
            lw = 3, color = "k")
    
    gl.plot(Klusters,mean_tr_ll + 2*std_tr_ll , color = "k", nf = 0, lw = 1, ls = "--", legend = ["Mean Train LL +- 2std"])
    gl.plot(Klusters,mean_tr_ll - 2*std_tr_ll , color = "k", nf = 0, lw = 1,ls = "--")
    gl.fill_between(Klusters, mean_tr_ll - 2*std_tr_ll, mean_tr_ll + 2*std_tr_ll, c = "k", alpha = 0.5)
    for i in range(len(logl_tr_CVs)):
        for k_i in range(len(Klusters)):
            gl.scatter(np.ones((len(logl_tr_CVs[i][k_i]),1))*Klusters[k_i], logl_tr_CVs[i][k_i], color = "k", alpha = 0.2, lw = 1)
        
    gl.plot(Klusters,mean_val_ll, nf = 0, color = "r",
            legend = ["Mean Validation LL (EM)"], lw = 3)
    gl.plot(Klusters,mean_val_ll + 2*std_val_ll , color = "r", nf = 0, lw = 1, ls = "--", legend = ["Mean Validation LL +- 2std"])
    gl.plot(Klusters,mean_val_ll - 2*std_val_ll , color = "r", nf = 0, lw = 1, ls = "--")
    gl.fill_between(Klusters, mean_val_ll - 2*std_val_ll, mean_val_ll + 2*std_val_ll, c = "r", alpha = 0.1)
    
    for i in range(len(logl_tr_CVs)):
        for k_i in range(len(Klusters)):
            gl.scatter(np.ones((len(logl_val_CVs[i][k_i]),1))*Klusters[k_i], logl_val_CVs[i][k_i], color = "r", alpha = 0.5, lw = 1)
    
    gl.savefig(folder_images + 'EM_Gaussian_CV_artificial_data.png', 
           dpi = 100, sizeInches = [12, 6])
