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

distribution_graph = 0
estimation_days_graph = 0
t_distribution_graph = 0
chi2_distribution_graph = 1
t_StatSig_graph = 0
chi2_StatSig_graph = 0
t_CI_graph = 0
chi2_CI_graph = 0

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

    gl.init_figure()
    ax1 = gl.scatter(ret1, np.zeros(ret1.shape), alpha = 0.5, lw = 4, AxesStyle = "Normal",
               labels = ["",symbolIDs[0], ""],
               legend = ["%i points"%ret1.size])
    
    for i in range(ret1.size/26):
        gl.scatter(ret1[i*26:(i+1)*26], np.ones(ret1[i*26:(i+1)*26].shape)*(i+1), alpha = 0.5, lw = 4, AxesStyle = "Normal",
                   legend = ["Day %i"%(i+1)])
    gl.set_zoom(ax = ax1, X = ret1,xlimPad = [0.1,0.8])

    gl.savefig(folder_images +'InitPointsInferenceDays.png', 
               dpi = 100, sizeInches = [10, 4])
    
if (estimation_days_graph):
    gl.init_figure()
    
    x_grid, y_values = bMA.gaussian1D_points(X = ret1, num = 100, std_K = 2, x_grid = None)
    
    ax1 = gl.plot(x_grid, y_values, alpha = 0.1, lw = 4, AxesStyle = "Normal",
               labels = ["",symbolIDs[0], "Distribution"],
               legend = ["%i points"%ret1.size] , color = "k")
              
    for i in range(ret1.size/26):
        D = ret1[i*26:(i+1)*26]
        x_grid, y_values = bMA.gaussian1D_points(X = D, num = 100, std_K = 2, x_grid = None)
        
        color = gl.get_color()
        gl.scatter(D, np.zeros(D.shape), alpha = 0.3, lw = 4, AxesStyle = "Normal",
                   legend = ["Day %i"%(i+1)],color = color)

        gl.plot(x_grid, y_values, color = color, fill = 1, alpha = 0.1)
                   
    gl.set_zoom(ax = ax1, X = ret1,xlimPad = [0.1,0.1])

    gl.savefig(folder_images +'InitPointsInferenceDaysEstimation.png', 
               dpi = 100, sizeInches = [10, 4])
               

if (t_distribution_graph):
    gl.init_figure()
    x_grid = np.linspace(-4,4,100)
    dfs = [1,3,5,26]
    
              
    for df in dfs:
        t_pdf = t.pdf(x_grid, df) 

        color = gl.get_color()
        ax1 = gl.plot(x_grid, t_pdf, alpha = 1, lw = 3, AxesStyle = "Normal",
                   legend = ["df %i"%df],color = color,
                labels = ["t-distribution","t","pdf(t)"])

    color = "k";
    x_grid, y_values = bMA.gaussian1D_points(mean = 0, std = 1, num = 100, x_grid = x_grid)        
    gl.plot(x_grid, y_values, alpha = 0.1, lw = 3, AxesStyle = "Normal",
               legend = ["Guassian"],color = color, fill = 1)

    gl.set_zoom(ax = ax1, X = x_grid,xlimPad = [0.1,0.1])

    gl.savefig(folder_images +'t-distribution.png', 
               dpi = 100, sizeInches = [14,6])
               
if (chi2_distribution_graph):
    
    x_grid = np.linspace(0,30,1000)
    dfs = [2,3,5,10,20]
    
    gl.init_figure()
    
    ################## 1st plot ###########################
    for df in dfs:
        
        chi2_pdf = chi2.pdf(x_grid, df) 
        color = gl.get_color()
        ax1 = gl.plot(x_grid, chi2_pdf, alpha = 1, lw = 3, AxesStyle = "Normal",
                   legend = ["df %i"%df],color = color,
                labels = ["chi2-distribution","x","pdf(x)"])

    gl.set_zoom(ax = ax1, X = x_grid,xlim = [-1, 30])
    
    gl.savefig(folder_images +'chi2-distribution.png', 
               dpi = 100, sizeInches = [10,4])     
    
    x_grid = np.linspace(0,100,1000)
    dfs = [2,3,5,10,20,50]
    
    gl.init_figure()
    for df in dfs:
        chi2_pdf = chi2.pdf(x_grid, df) 
        color = gl.get_color()
        ax2 = gl.plot(x_grid/df, df* chi2_pdf, alpha = 1, lw = 3, AxesStyle = "Normal",
                   legend = ["df %i"%df],color = color,
                labels = ["Scaled chi2-distribution2","x/df","pdf(x/df)"])
                
#    x_grid, y_values = bMA.gaussian1D_points(mean = 0, std = 1, num = 100, x_grid = x_grid)        
#    gl.plot(x_grid, y_values, alpha = 1, lw = 3, AxesStyle = "Normal",
#               legend = ["Guassian"],color = color)

    gl.set_zoom(ax = ax2, X = x_grid,xlim = [-0.2,3])

    gl.savefig(folder_images +'scaled_chi2-distribution.png', 
               dpi = 100, sizeInches = [10,4])     
               
if(t_StatSig_graph):

    x_grid_all = np.linspace(-4,4,100)
    dfs = [25]
    
    mu_0 = 0;
    ## Draw the distribution          
    for df in dfs:
        t_pdf = t.pdf(x_grid_all, df) 

        color = gl.get_color()
        ax1 = gl.plot(x_grid_all, t_pdf, alpha = 1, lw = 3, AxesStyle = "Normal",
                   legend = ["df %i"%df],color = color,
                labels = ["double-sided event","|t|","pdf(t)"])
    
    ## Draw the p-value for each day.
    days = [0,1,2,3,4]
    
    ax1 = None
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*26]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        t_val = (mu_hat - mu_0)/(S/np.sqrt(N))
        t_val_abs = np.abs(t_val)
        x_grid = np.linspace(t_val_abs,4,100)
        t_pdf = t.pdf(x_grid, N-1) 
        p_val = 1 - t.cdf(t_val_abs, N-1)
        p_val = 2 * (1 - t.cdf(t_val_abs, N-1))
        color = gl.get_color()
        gl.scatter(t_val_abs, t.pdf(t_val_abs, N-1) , color = color,
                legend = ["T: %.2f, p-v: %.3f"%(t_val,p_val)])
        
        gl.plot(x_grid, t_pdf, color = color, fill = 1, alpha = 0.2)
               

    gl.set_zoom(ax = ax1, X = x_grid_all,xlimPad = [0.1,0.1])
    
    gl.savefig(folder_images +'t-StatSig.png', 
               dpi = 100, sizeInches = [14,6])
               
    
    ##### Plot the both sided one !!!
    gl.init_figure()
    ax0 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    t_pdf = t.pdf(x_grid_all, df) 
    gl.plot(x_grid_all, t_pdf, alpha = 1, lw = 3, AxesStyle = "Normal",
                       legend = ["df %i"%df],color = color,
                    labels = ["right-tail event","t","pdf(t)"])
    ## Plot the right sided
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*26]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        t_val = (mu_hat - mu_0)/(S/np.sqrt(N))
        x_grid = np.linspace(t_val,4,100)
        
        t_pdf = t.pdf(x_grid, N-1) 
        p_val = 1 - t.cdf(t_val, N-1)
        
        color = gl.get_color()
        gl.scatter(t_val, t.pdf(t_val, N-1) , color = color,
                legend = ["T: %.2f, p-v: %.3f"%(t_val,p_val)], ax= ax0)
        
        gl.plot(x_grid, t_pdf, color = color, fill = 1, alpha = 0.2, ax= ax0)
               
    ## Plot the right sided
    ax1 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1, sharex = ax0, sharey = ax0)
    t_pdf = t.pdf(x_grid_all, df) 
    gl.plot(x_grid_all, t_pdf, alpha = 1, lw = 3, AxesStyle = "Normal",
                       legend = ["df %i"%df],color = color,
                    labels = ["left-tail event","t","pdf(t)"])
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*26]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        t_val = (mu_hat - mu_0)/(S/np.sqrt(N))
        
        x_grid = np.linspace(-4,t_val,100)
        t_pdf = t.pdf(x_grid, N-1) 
        p_val = t.cdf(t_val, N-1)
        
        color = gl.get_color()
        gl.scatter(t_val, t.pdf(t_val, N-1) , color = color,
                legend = ["T: %.2f, p-v: %.3f"%(t_val,p_val)], ax= ax1)
        
        gl.plot(x_grid, t_pdf, color = color, fill = 1, alpha = 0.2, ax= ax1)
               
    gl.set_zoom(ax = ax1, X = x_grid_all,xlimPad = [0.1,0.1])
    
    gl.savefig(folder_images +'t-StatSig2.png', 
               dpi = 100, sizeInches = [14,6])
               
    
if(chi2_StatSig_graph):

#    gl.init_figure()
    days = [0,1,2,3,4]
    chi2_left = 13.12  # 0.025
    chi2_right = 40.65 # 9.775
    N = 26
    sigma2_w = np.std(ret1[days[0]*N:(days[-1] +1)*N])**2

    gl.init_figure()
    
    ### PLot the right handed !!!
    ax0 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    
    x_grid = np.linspace(5,50,100)
    x_grid_CI = np.linspace(chi2_left,chi2_right,100)
    
    chi2_pdf = chi2.pdf(x_grid, N-1) 

    gl.plot(x_grid, chi2_pdf, sharex = ax1, color = color,
            legend = ["N: %i"%(N)],  labels = ["right-sided p-value","chi2","pdf(chi2)"])
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*N]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        chi2_val = (N-1)*S**2/sigma2_w
        chi2_val_pdf = chi2.pdf(chi2_val, N-1)
        p_val = 1 - chi2.cdf(chi2_val, N-1)

        x_grid = np.linspace(chi2_val,50,100)
        chi2_pdf = chi2.pdf(x_grid, N-1) 
        
        color = gl.get_color()
        gl.scatter(chi2_val, chi2_val_pdf , color = color,
                legend = ["T: %.2f, p-v: %.3f"%(chi2_val,p_val)])
        gl.plot(x_grid, chi2_pdf, color = color, fill = 1, alpha = 0.2)


    ### Plot the left handed !!
    ax1 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
    
    x_grid = np.linspace(5,50,100)
    x_grid_CI = np.linspace(chi2_left,chi2_right,100)
    
    chi2_pdf = chi2.pdf(x_grid, N-1) 

    gl.plot(x_grid, chi2_pdf, sharex = ax1, color = color,
            legend = ["N: %i"%(N)],  labels = ["left-sided p-value","chi2","pdf(chi2)"])
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*N]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        chi2_val = (N-1)*S**2/sigma2_w
        chi2_val_pdf = chi2.pdf(chi2_val, N-1)
        p_val =  chi2.cdf(chi2_val, N-1)

        x_grid = np.linspace(5, chi2_val,100)
        chi2_pdf = chi2.pdf(x_grid, N-1) 
        
        color = gl.get_color()
        gl.scatter(chi2_val, chi2_val_pdf , color = color,
                legend = ["T: %.2f, p-v: %.3f"%(chi2_val,p_val)])
        gl.plot(x_grid, chi2_pdf, color = color, fill = 1, alpha = 0.2)

    gl.savefig(folder_images +'chi2-StatSig.png', 
               dpi = 100, sizeInches = [14,6])


    
    ### PLot the both sided handed !!!
    gl.init_figure()
    
    x_grid = np.linspace(5,50,100)
    x_grid_CI = np.linspace(chi2_left,chi2_right,100)
    
    chi2_pdf = chi2.pdf(x_grid, N-1) 

    gl.plot(x_grid, chi2_pdf, sharex = ax1, color = color,
            legend = ["N: %i"%(N)],  labels = ["both-sided p-value","chi2","pdf(chi2)"])
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*N]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        chi2_val = (N-1)*S**2/sigma2_w
        chi2_val_pdf = chi2.pdf(chi2_val, N-1)
        
        p_val1 = 1 - chi2.cdf(chi2_val, N-1)
        p_val2 = chi2.cdf(chi2_val, N-1)
        if (p_val1 > p_val2):  # left handed
            p_val = 2* p_val2
            x_grid = np.linspace(chi2_val,50,100)
            side = "l"
        else:
            p_val = 2* p_val1
            x_grid = np.linspace(5,chi2_val,100)
            side = "r"
        ##  
        
        chi2_pdf = chi2.pdf(x_grid, N-1) 
        
        color = gl.get_color()
        gl.scatter(chi2_val, chi2_val_pdf , color = color,
                legend = ["T: %.2f, p-v (%s): %.3f"%(chi2_val,side,p_val)])
        gl.plot(x_grid, chi2_pdf, color = color, fill = 1, alpha = 0.2)

    gl.savefig(folder_images +'chi2-StatSig2.png', 
               dpi = 100, sizeInches = [14,6])
    
if(t_CI_graph):
#    gl.init_figure()
    gl.set_subplots(1,2)
    
    t_int = 2.06
    
    days = [0,1]
    
    ax1 = None
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*26]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        x_grid = np.linspace(-4,4,100)
        x_grid_CI = np.linspace(-t_int,t_int,100)
        
        t_pdf = t.pdf(x_grid, N-1) 
        t_pdf_CI = t.pdf(x_grid_CI, N-1)   
        # Transform from t to mu.
        x_grid = -x_grid*S + mu_hat
        x_grid_CI = -x_grid_CI*S + mu_hat
        mu_left = -t_int*S + mu_hat
        mu_right = t_int*S + mu_hat
        
        color = gl.get_color()
        ax1 = gl.scatter(D, np.zeros(D.shape), sharex = ax1, alpha = 0.3, lw = 4, AxesStyle = "Normal",
                   legend = ["Day %i"%(i+1)],color = color, nf = 1,
                    labels = ["CI of mu in Day %i"%(i+1),"mu","pdf(mu)"])
    
        gl.plot(x_grid, t_pdf, color = color,
                legend = ["mu: %.2e, S: %.2e"%(mu_hat,S)])
        
        gl.plot(x_grid_CI, t_pdf_CI, color = color, fill = 1, alpha = 0.1,
                legend = ["CI = [%.2e,%.2e]"%(mu_left, mu_right)]) 
        
        #gl.set_zoom(ax = ax1, X = ret1,xlimPad = [0.1,0.1])
    
    gl.savefig(folder_images +'t-CI.png', 
               dpi = 100, sizeInches = [14,6])
           

if(chi2_CI_graph):
#    gl.init_figure()
    gl.set_subplots(1,2)
    
    chi2_left = 13.12  # 0.025
    chi2_right = 40.65 # 9.775
    
    days = [0,1]
    
    ax1 = None
    for i in days:
    ## Day 1    
        D = ret1[i*26:(i+1)*26]
        N,mu_hat,S = D.size,np.mean(D),np.std(D)
        
        x_grid = np.linspace(5,50,100)
        x_grid_CI = np.linspace(chi2_left,chi2_right,100)
        
        chi2_pdf = chi2.pdf(x_grid, N-1) 
        chi2_pdf_CI = chi2.pdf(x_grid_CI, N-1)   
        # Transform from t to mu.
        x_grid = (N-1)*S**2/x_grid
        x_grid_CI = (N-1)*S**2/x_grid_CI
        
        sigma_left = (N-1)*S**2/chi2_left
        sigma_right =(N-1)*S**2/chi2_right
        
        color = gl.get_color()

        ax1 = gl.plot(x_grid, chi2_pdf, sharex = ax1, color = color,nf = 1,
                legend = ["S: %.2e"%(S)],  labels = ["CI of sigma in Day %i"%(i+1),"sigma","pdf(sigma)"])
        
        gl.plot(x_grid_CI, chi2_pdf_CI, color = color, fill = 1, alpha = 0.1,
                legend = ["CI = [%.2e,%.2e]"%(sigma_left, sigma_right)]) 
        
        #gl.set_zoom(ax = ax1, X = ret1,xlimPad = [0.1,0.1])

    gl.savefig(folder_images +'chi2-CI.png', 
               dpi = 100, sizeInches = [14,6])
           
           