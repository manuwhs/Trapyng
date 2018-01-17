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
import MultivariateStatLib as MSlib
import indicators_lib as indl
import DDBB_lib as DBl
plt.close("all")

folder_images = "../pics/Trapying/MultivariateStat/"
##############################################
########## FLAGS ############################

use_real_data_flag = 0
plot_explained_var = 0
plot_projections_2Assets = 0
plot_projections_AllAssets = 0
plot_reconstruction_2D = 1
plot_Component_Pattern = 0 
plot_Component_Pattern_Profile = 0

##########################################################################
################# DATA OBTAINING ######################################
##########################################################################

if (use_real_data_flag == 1):
    mus = np.array([-0.5,-1,1.5])
    stds = np.array([1,1.5,2])
    Nsam = 1000
    Nx = 10
    
    if (Nx >3):
        mus = np.random.randn(Nx)
        stds = np.random.randn(Nx)
        
    X = []
    for i in range(Nx):
        X_i = np.random.randn(Nsam,1)*stds[i] + mus[i]
        X.append(X_i)
    
    X = np.concatenate((X),axis = 1)
    
else:
    Nx = 25
    dataSource =  "Google"  # Hanseatic  FxPro GCI Yahoo
    [storage_folder, info_folder, 
     updates_folder] = ul.get_foldersData(source = dataSource)
    folder_images = "../pics/Trapying/MultivariateStat/"
    ######## SELECT SYMBOLS AND PERIODS ########
    periods = [5]
    
    # Create porfolio and load data
    cmplist = DBl.read_NASDAQ_companies(whole_path = "../storage/Google/companylist.csv")
    cmplist.sort_values(by = ['MarketCap'], ascending=[0],inplace = True)
    symbolIDs = cmplist.iloc[0:Nx]["Symbol"].tolist()
    for period in periods:
        Cartera = CPfl.Portfolio("BEST_PF", symbolIDs, [period]) 
        Cartera.set_csv(storage_folder)
    sdate = dt.datetime.strptime("6-8-2017", "%d-%m-%Y")
    edate = dt.datetime.strptime("9-8-2017", "%d-%m-%Y")
    Cartera.set_interval(sdate, edate)
    opentime, closetime = Cartera.get_timeData(symbolIDs[0],periods[0]).guess_openMarketTime()
    dataTransform = ["intraday", opentime, closetime]
    
    #Cartera.get_timeData(symbolIDs[0],periods[0]).fill_data()
    #Cartera.get_timeData(symbolIDs[1],periods[0]).fill_data()
    symbolIDs = Cartera.get_symbolIDs()
    X = Cartera.get_timeSeriesReturn(symbolIDs,periods[0])
    Nsam = 130
    for i in range(len(X)):
        X[i] = X[i][:Nsam]   # Truncate the values since some of them hace 1 more sample
        print (X[i].shape)
    X = np.concatenate(X,axis = 1) * 100
    dates = Cartera.get_timeData(symbolIDs[1],periods[0]).get_dates()
    symbolIDs = Cartera.get_symbolIDs()
    
    AAPL_id = np.where(np.array(symbolIDs) == 'AAPL')[0][0]
    GOOGL_id = np.where(np.array(symbolIDs) == "GOOGL")[0][0]
############################################################
################# PLOT DATA ###############################
############################################################
    

if (plot_explained_var):
    ## Get the 2D projections. Only using 2 assets so that we see the rotation
    explained_var = MSlib.get_explained_var_PCA(X, np.zeros((Nsam,1)))
    explained_var_ratio = explained_var / np.sum(explained_var)
    
    cumvarPCA = np.cumsum(explained_var_ratio)
    
    gl.set_subplots(1,2)
    gl.plot([],explained_var,lw=3, nf = 1,
            labels= ['Explained Variance Eigenvalues', 'Number of components', 'Explained Variance Eigenvalues'])
    
    gl.plot([],cumvarPCA,lw=3, nf = 1,
            labels= ['Cumulative Explained Variance Ratio', 'Number of components', ' Cumulative Explained Variance Ratio'])
    
    gl.savefig(folder_images +'ExplainedVarPCA.png', 
               dpi = 100, sizeInches = [14, 7])
    
if (plot_projections_2Assets):
    X2D = X[:,[AAPL_id, GOOGL_id]]
    componentsPCA =  MSlib.get_components_PCA(X2D, n_components = 2)
    Xproj = MSlib.get_projected_PCA(X2D, n_components = 2)

    gl.init_figure()
    X_1,X_2 = X2D[:,[0]], X2D[:,[1]]
    mu_1, mu_2 = np.mean(Xproj, axis =0)
    std_1,std_2 = np.std(Xproj,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax0 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax0, lw = 4, AxesStyle = "Normal",
               labels = ["Original 2D data","X1: %s" %(symbolIDs[AAPL_id]), "X2:  %s" %(symbolIDs[GOOGL_id])])
    
    ax0.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(X2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(X2D)
    cov = bMA.get_covMatrix(X2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(X2D)
    r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax0, ls = "--",color = "k", lw = 2,
             legend = ["Corr: %.2f"%(corr[0,1])])
    
    gl.plot([mean[0], mean[0] + vecs[0,0]*w], 
            [mean[1], mean[1] + vecs[0,1]*w], ax = ax0, ls = "--",color = "k")
    gl.plot([mean[0], mean[0] + vecs[1,0]*h], 
            [mean[1], mean[1] + vecs[1,1]*h], ax = ax0, ls = "--",color = "k")
    
    gl.plot([mean[0], mean[0] - vecs[0,0]*w], 
            [mean[1], mean[1] - vecs[0,1]*w], ax = ax0, ls = "--",color = "k")
    gl.plot([mean[0], mean[0] - vecs[1,0]*h], 
            [mean[1], mean[1] - vecs[1,1]*h], ax = ax0, ls = "--",color = "k")
            
    ax0.axis('equal')
    gl.set_zoom(ax = ax0, X =r_ellipse[:,0], Y = r_ellipse[:,1],
                ylimPad = [0.2,0.2],xlimPad = [0.2,0.2])

    vecs_original = vecs
    
    #### PLOT THE TRANSFORMED ONES !!!
    ##############################################################################
    X_1,X_2 = Xproj[:,[0]], Xproj[:,[1]]
    mu_1, mu_2 = np.mean(Xproj, axis =0)
    std_1,std_2 = np.std(Xproj,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax1 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1,  sharex = ax0, sharey = ax0)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["PCA Prejected 2D data","Y1", "Y2"], color = "dark navy blue")
    
    ax1.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(Xproj, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(Xproj)
    cov = bMA.get_covMatrix(Xproj)
    vecs,vals = bMA.get_eigenVectorsAndValues(Xproj)
    r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2,
             legend = ["Corr: %.2f"%(corr[0,1])])
    
    gl.plot([mean[0], mean[0] + vecs[0,0]*w], 
            [mean[1], mean[1] + vecs[0,1]*w], ax = ax1, ls = "--",color = "k",
            legend = ["Y1 = %0.2f X_1 + %0.2f X_2"%(vecs_original[0,0], vecs_original[0,1])])
    gl.plot([mean[0], mean[0] + vecs[1,0]*h], 
            [mean[1], mean[1] + vecs[1,1]*h], ax = ax1, ls = "--",color = "k",
            legend = ["Y2 = %0.2f X_1 + %0.2f X_2"%(vecs_original[1,0], vecs_original[1,1])])
    
    gl.plot([mean[0], mean[0] - vecs[0,0]*w], 
            [mean[1], mean[1] - vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
    gl.plot([mean[0], mean[0] - vecs[1,0]*h], 
            [mean[1], mean[1] - vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
            
    ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
    
    gl.set_zoom(ax = ax1, X =r_ellipse[:,0], Y = r_ellipse[:,1],
                ylimPad = [0.2,0.2],xlimPad = [0.2,0.2])


    gl.savefig(folder_images +'PCA_2D.png', 
               dpi = 100, sizeInches = [18, 7])



if (plot_reconstruction_2D):
    
    gl.init_figure()
    
    ################### PLOT THE ORIGINAL ONES ###########333
    
    X2D = X[:,[AAPL_id, GOOGL_id]]
    componentsPCA =  MSlib.get_components_PCA(X2D, n_components = 2)
    Xproj = MSlib.get_projected_PCA(X2D, n_components = 2)


    X_1,X_2 = X2D[:,[0]], X2D[:,[1]]
    mu_1, mu_2 = np.mean(Xproj, axis =0)
    std_1,std_2 = np.std(Xproj,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax0 = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=1)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax0, lw = 4, AxesStyle = "Normal",
               labels = ["Original 2D data","X1: %s" %(symbolIDs[AAPL_id]), "X2:  %s" %(symbolIDs[GOOGL_id])])
    
    ax0.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(X2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(X2D)
    cov = bMA.get_covMatrix(X2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(X2D)
    r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax0, ls = "--",color = "k", lw = 2,
             legend = ["Corr: %.2f"%(corr[0,1])])
    
    gl.plot([mean[0], mean[0] + vecs[0,0]*w], 
            [mean[1], mean[1] + vecs[0,1]*w], ax = ax0, ls = "--",color = "k")
    gl.plot([mean[0], mean[0] + vecs[1,0]*h], 
            [mean[1], mean[1] + vecs[1,1]*h], ax = ax0, ls = "--",color = "k")
    
    gl.plot([mean[0], mean[0] - vecs[0,0]*w], 
            [mean[1], mean[1] - vecs[0,1]*w], ax = ax0, ls = "--",color = "k")
    gl.plot([mean[0], mean[0] - vecs[1,0]*h], 
            [mean[1], mean[1] - vecs[1,1]*h], ax = ax0, ls = "--",color = "k")
            
    ax0.axis('equal')
    gl.set_zoom(ax = ax0, X =r_ellipse[:,0], Y = r_ellipse[:,1],
                ylimPad = [0.2,0.2],xlimPad = [0.2,0.2])


    #### PLOT THE TRANSFORMED ONES !!!
    ##############################################################################
    Ny = 20
    componentsPCA =  MSlib.get_components_PCA(X, n_components = Ny)
    Y = MSlib.get_projected_PCA(X, n_components = Ny)
    Xrec = componentsPCA.T.dot(Y.T)
    
    i_1, i_2 = AAPL_id,GOOGL_id
    Xrec = Xrec.T
    X_1,X_2 = Xrec[:,[i_1]], Xrec[:,[i_2]]
    Xrec2D = np.concatenate(( X_1,X_2), axis = 1)
    mu_1, mu_2 = np.mean(Xrec2D, axis =0)
    std_1,std_2 = np.std(Xrec2D,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax1 = gl.subplot2grid((1,4), (0,1), rowspan=1, colspan=1, sharex = ax0, sharey = ax0)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["m = %i"%Ny,"Xrec1", "Xrec2"])
    
    ax1.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(Xrec2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(Xrec2D)
    cov = bMA.get_covMatrix(Xrec2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(Xrec2D)
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

    ##############################################################################
    Ny = 12
    componentsPCA =  MSlib.get_components_PCA(X, n_components = Ny)
    Y = MSlib.get_projected_PCA(X, n_components = Ny)
    Xrec = componentsPCA.T.dot(Y.T)
    
    i_1, i_2 = AAPL_id,GOOGL_id
    Xrec = Xrec.T
    X_1,X_2 = Xrec[:,[i_1]], Xrec[:,[i_2]]
    Xrec2D = np.concatenate(( X_1,X_2), axis = 1)
    mu_1, mu_2 = np.mean(Xrec2D, axis =0)
    std_1,std_2 = np.std(Xrec2D,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax1 = gl.subplot2grid((1,4), (0,2), rowspan=1, colspan=1, sharex = ax0, sharey = ax0)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["m = %i"%Ny,"Xrec1", "Xrec2"])
    
    ax1.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(Xrec2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(Xrec2D)
    cov = bMA.get_covMatrix(Xrec2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(Xrec2D)
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

    ##############################################################################
    Ny = 5
    componentsPCA =  MSlib.get_components_PCA(X, n_components = Ny)
    Y = MSlib.get_projected_PCA(X, n_components = Ny)
    Xrec = componentsPCA.T.dot(Y.T)
    
    i_1, i_2 = AAPL_id,GOOGL_id
    Xrec = Xrec.T
    X_1,X_2 = Xrec[:,[i_1]], Xrec[:,[i_2]]
    Xrec2D = np.concatenate(( X_1,X_2), axis = 1)
    mu_1, mu_2 = np.mean(Xrec2D, axis =0)
    std_1,std_2 = np.std(Xrec2D,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax1 = gl.subplot2grid((1,4), (0,3), rowspan=1, colspan=1, sharex = ax0, sharey = ax0)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["m = %i"%Ny,"Xrec1", "Xrec2"])
    
    ax1.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(Xrec2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(Xrec2D)
    cov = bMA.get_covMatrix(Xrec2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(Xrec2D)
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
    
    gl.savefig(folder_images +'PCA_reconstruct.png', 
               dpi = 100, sizeInches = [18, 4])


if ( plot_projections_AllAssets):
    componentsPCA =  MSlib.get_components_PCA(X, n_components = 20)
    Xproj = MSlib.get_projected_PCA(X, n_components = 20)

    gl.init_figure()
    #### PLOT THE TRANSFORMED ONES !!!
    ##############################################################################
    i_1, i_2 = 0,1
    X_1,X_2 = Xproj[:,[i_1]], Xproj[:,[i_2]]
    X_proj2D = np.concatenate(( X_1,X_2), axis = 1)
    mu_1, mu_2 = np.mean(X_proj2D, axis =0)
    std_1,std_2 = np.std(X_proj2D,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["PCA Prejected 2D data","Y%i"%(i_1 +1), "Y%i"%(i_2+1)])
    
    ax1.axis('equal')
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(X_proj2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(X_proj2D)
    cov = bMA.get_covMatrix(X_proj2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(X_proj2D)
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


    ####################### Second Axis ############################################
    i_1, i_2 = 2,8
    X_1,X_2 = Xproj[:,[i_1]], Xproj[:,[i_2]]
    X_proj2D = np.concatenate(( X_1,X_2), axis = 1)
    mu_1, mu_2 = np.mean(X_proj2D, axis =0)
    std_1,std_2 = np.std(X_proj2D,axis =0)
    
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    ## Do stuff now
    ax1 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
              labels = ["PCA Prejected 2D data","Y%i"%(i_1 + 1), "Y%i"%(i_2 + 1)]
              , color = "dark navy blue")
    
    ################# Draw the error ellipse  #################
    mean,w,h,theta = bMA.get_gaussian_ellipse_params(X_proj2D, Chi2val = 2.4477)
    corr = bMA.get_corrMatrix(X_proj2D)
    cov = bMA.get_covMatrix(X_proj2D)
    vecs,vals = bMA.get_eigenVectorsAndValues(X_proj2D)
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
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
    
    gl.set_zoom(ax = ax1, X =r_ellipse[:,0], Y = r_ellipse[:,1],
                ylim = [-4, 4.5],xlim = [-2.5,5.6])

    gl.savefig(folder_images +'PCA_All.png', 
               dpi = 100, sizeInches = [18, 7])
    
    
    
if (plot_Component_Pattern):
    componentsPCA =  MSlib.get_components_PCA(X, n_components = 20)
    Xproj = MSlib.get_projected_PCA(X, n_components = 20)

    #### PLOT THE TRANSFORMED ONES !!!
    ##############################################################################
    gl.init_figure()
    
    Nwindows = 3
    sel = [[0,1],[2,4], [3,8]]
    for nw in range(Nwindows):
        
        ## Compute the data
        i_1, i_2 = sel[nw]
        X_1,X_2 = Xproj[:,[i_1]], Xproj[:,[i_2]]
        X_proj2D = np.concatenate(( X_1,X_2), axis = 1)
        mu_1, mu_2 = np.mean(X_proj2D, axis =0)
        std_1,std_2 = np.std(X_proj2D,axis =0)
        
        correlations_1 = []
        correlations_2 = []
        
        Nshow = 6
        for i in range(Nshow):
            corr1 = np.corrcoef(np.concatenate(( X_1,X[:,[i]]), axis = 1).T)[0,1]
            corr2 = np.corrcoef(np.concatenate(( X_2,X[:,[i]]), axis = 1).T)[0,1]
            correlations_1.append(corr1)
            correlations_2.append(corr2)
        
        ax1 = gl.subplot2grid((1,Nwindows), (0,nw), rowspan=1, colspan=1)
        
        for i in range(Nshow): # For each correlation to show
            gl.scatter(correlations_1[i],correlations_2[i], alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal", color = "k",
                       labels = ["PCA Prejected 2D data","P%i"%(i_1 +1), "P%i"%(i_2+1)])
            
            gl.plot([0, correlations_1[i]], [0, correlations_2[i]], color = "k")
        
        ## Anotate the companies !!
        i = 0;
        for xy in zip(correlations_1, correlations_2):                                       # <--
            ax1.annotate(symbolIDs[i], xy=xy, textcoords='data')
            i += 1;
        
        ## Plot the circle !
        r_ellipse = bMA.get_ellipse_points([0,0],1,1,0)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2)
        
        ax1.axis('equal')
        
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
  
    gl.savefig(folder_images +'PCA_CompPattern.png', 
               dpi = 100, sizeInches = [18, 6])

if (plot_Component_Pattern_Profile):
    Ny = 10
    Nx = 7
    
    componentsPCA =  MSlib.get_components_PCA(X, n_components = Ny)
    Xproj = MSlib.get_projected_PCA(X, n_components = Ny)

    #### PLOT THE TRANSFORMED ONES !!!
    ##############################################################################
    corr = np.corrcoef(X[:,range(Nx)].T,Xproj.T)
    corrXY = corr[range(Nx,Ny+ Nx),:]
    corrXY = corrXY[:,range(Nx)]
    
    for i in range(Ny):
        gl.plot(symbolIDs[:Nx],corrXY[i,:], labels = ["Component Pattern Profile Plot", "","Corr(X_i,Y_j)"],
                legend = ["Y_%i"%(i+1)]);

    gl.savefig(folder_images +'PCA_CompPatternProfile.png', 
               dpi = 100, sizeInches = [12, 4])
    
    gl.init_figure()
    for i in range(Ny):
        gl.plot(symbolIDs[:Nx],corrXY[i,:]*corrXY[i,:], labels = ["Square Component Pattern Profile Plot", "","Corr(X_i,Y_j)"],
                legend = ["Y_%i"%(i+1)]);

    gl.savefig(folder_images +'PCA_CompPatternProfile2.png', 
               dpi = 100, sizeInches = [12, 4])
    
    #### PLOT THE TRANSFORMED ONES !!!
    ##############################################################################
    gl.init_figure()
    for i in range(Ny):
        gl.plot(symbolIDs[:Ny], componentsPCA[i,:Nx], labels = ["Projection coefficients", "","p_ij"],
                legend = ["Y_%i"%(i+1)]);
                
    gl.init_figure()
    for i in range(Ny):
        gl.plot(symbolIDs[:Ny], componentsPCA[i,:Nx]* componentsPCA[i,:Nx], labels = ["Projection coefficients", "","p_ij"],
                legend = ["Y_%i"%(i+1)]);
                
    gl.savefig(folder_images +'PCA_CompPat.png', 
               dpi = 100, sizeInches = [12, 4])
    # Each row is the correlation between the X_i and all the Y_j, j = 1,...,Ny
#    ax1 = gl.subplot2grid((1,Nwindows), (0,nw), rowspan=1, colspan=1)
#    
#    for i in range(Nshow): # For each correlation to show
#        gl.scatter(correlations_1[i],correlations_2[i], alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal", color = "k",
#                   labels = ["PCA Prejected 2D data","P%i"%(i_1 +1), "P%i"%(i_2+1)])
#        
#        gl.plot([0, correlations_1[i]], [0, correlations_2[i]], color = "k")
#    
#    ## Anotate the companies !!
#    i = 0;
#    for xy in zip(correlations_1, correlations_2):                                       # <--
#        ax1.annotate(symbolIDs[i], xy=xy, textcoords='data')
#        i += 1;
#    
#    ## Plot the circle !
#    r_ellipse = bMA.get_ellipse_points([0,0],1,1,0)
#    gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2)
#    
#    ax1.axis('equal')
#        
#    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
#  
#    gl.savefig(folder_images +'PCA_CompPatternProfile.png', 
#               dpi = 100, sizeInches = [18, 6])
    
#alpha_val = 1
#classColors =  ["r","k"] # np.random.rand(Nclasses,3)
#    # If we are given labels, we can plot in different colors the different classes.
#if(type(labels) != type(None)):
#    
#labelsTrain = np.array(y_train)
## View the  original data and the projection vectors
#labels = np.unique(labelsTrain.astype(np.int))
#plt.figure()
#for i,l in enumerate(labels):
#    plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=alpha_val,c=classColors[i])
#plt.title('First 2 components of projected data')
#plt.show()
#    