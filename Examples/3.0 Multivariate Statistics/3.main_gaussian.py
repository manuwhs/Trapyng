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

folder_images = "../pics/Trapying/MultivariateStat/"
##############################################
########## FLAGS ############################

distribution_graph = 1
distribution_graph_2D = 0
distribution_graph_3D =0
distribution_graph_3D_slices = 0

distribution_graph_3D_condtional = 1;
##########################################################################
################# DATA OBTAINING ######################################
##########################################################################
mus = np.array([-0.5,-1,1.5])
stds = np.array([1,1.5,2])
Nsam = 1000
Nx = 3

X = []
for i in range(Nx):
    X_i = np.random.randn(Nsam,1)*stds[i] + mus[i]
    X.append(X_i)

X = np.concatenate((X),axis = 1)
Nsim = 1000
x_grid = np.linspace(-6,8,Nsim)

if(distribution_graph):
    ## Plot the 3 distributions ! 
    gl.init_figure()
    for i in range(Nx):
        
        X_i = X[:,[i]]
        x_grid, y_values = bMA.gaussian1D_points(mean = mus[i], std = stds[i],
        x_grid = x_grid)
        
        color = gl.get_color()
        gl.scatter(X_i, np.zeros(X_i.shape), alpha = 0.1, lw = 4, AxesStyle = "Normal",
                   color = color, labels = ["3 independent Gaussian distributions","x","pdf(x)"])
                   
        gl.plot(x_grid, y_values, color = color, fill = 1, alpha = 0.1,
                legend = ["X%i: m:%.1f, std:%.1f"%(i+1,mus[i],stds[i])])  
        
        
        gl.savefig(folder_images +'Gaussians.png', 
           dpi = 100, sizeInches = [18, 10])
           
############################################################
################# PLOT DATA ###############################
############################################################

if(distribution_graph_2D):
    # Get the histogram and gaussian estimations !
    ## Scatter plot of the points 
#    gl.init_figure()
    i_1 = 2
    i_2 = 0
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mus[i_1],mus[i_2]
    std_1, std_2 = stds[i_1],stds[i_2]
    
    mu = mus[[i_1,i_2]]
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    
    ## Do stuff now
    ax1 = gl.subplot2grid((4,4), (1,0), rowspan=3, colspan=3)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["","X1", "X2"])
    
    ## X distribution
    ax2 = gl.subplot2grid((4,4), (0,0), rowspan=1, colspan=3, sharex = ax1)

    x_grid, y_val = bMA.gaussian1D_points(mean = mu_1, std  = std_1, std_K = std_K)
    gl.plot(x_grid, y_val, color = "k",
            labels = ["","",""], legend = ["M: %.1f, std: %.1f"%(mu_1, std_1)],
            AxesStyle = "Normal - No xaxis")
    
    # Y distribution
    ax3 = gl.subplot2grid((4,4), (1,3), rowspan=3, colspan=1,sharey = ax1,)

    x_grid, y_val = bMA.gaussian1D_points(mean = mu_2, std  = std_2, std_K = std_K)
    gl.plot(y_val, x_grid, color = "k",
            labels = ["","",""], legend = ["M: %.1f, std: %.1f"%(mu_2, std_2)],
            AxesStyle = "Normal - No yaxis")

    ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)
    
    
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=20j, ybins=20j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ax1.contour(xx, yy, zz, linewidths = 3, linestyles = "solid", alpha = 0.8,
                colors = None)
    
    gl.savefig(folder_images +'Gaussian_2DX.png', 
               dpi = 100, sizeInches = [18, 14])

if(distribution_graph_3D):
    i_1 = 2
    i_2 = 0
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mus[i_1],mus[i_2]
    std_1, std_2 = stds[i_1],stds[i_2]
    
    mu = mus[[i_1,i_2]]
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    
    ################ Contour plot of the scatter plot ################
    gl.init_figure()
    std_K= 3
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=30j, ybins=30j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ## Plot the 3D surface
    ax3D = gl.plot_3D(xx, yy, zz, nf = 0,
                      labels = ["2D Gaussian pdf","X_1","X_2"])
    
    ## Limits of the plotting !
    xmin,xmax = [np.min(xx.flatten()), np.max(xx.flatten())]
    ymin,ymax = [np.min(yy.flatten()), np.max(yy.flatten())]
    zmin,zmax = [np.min(zz.flatten()), np.max(zz.flatten())]
    
    xmin = xmin - (xmax - xmin)*0.2
    xmax = xmax + (xmax - xmin)*0.2
    ymin = ymin - (ymax - ymin)*0.2
    ymax = ymax + (ymax - ymin)*0.2
    
    xymin, xymax = [np.min([xmin,ymin]),np.max([xmax,ymax])]
    
    # Plot the marginalization of X
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_1, std  = std_1, std_K = std_K)
    y_val = y_val * (1/np.sqrt(2*np.pi*std_2*std_2))
    
    ax3D.plot(x_grid,y_val,xymax, zdir='y')
    ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),xymax, zdir='y')
  
    # Plot the marginalization of Y
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_2, std  = std_2, std_K = std_K) 
    y_val = y_val * (1/np.sqrt(2*np.pi*std_1*std_1))
    ax3D.plot(x_grid,y_val, xymin, zdir='x')
    ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),xymin, zdir='x')
    
    # Plot the contour lines:
    ax3D.contour(xx, yy, zz, offset=0, zdir='z')

    # Set the visualization limits !
#    ax3D.set_xlim(xmin, xmax)
#    ax3D.set_ylim(ymin, ymax)
    
    ax3D.set_xlim(xymin, xymax)
    ax3D.set_ylim(xymin, xymax)
    ax3D.set_zlim(zmin, zmax)
    
#    ax1.pcolormesh(xx, yy, zz)
#    ax1.imshow(zz, origin='lower', extent=[-3,3,-3,3], cmap="gray")

    gl.savefig(folder_images +'Gaussian3D.png', 
               dpi = 100, sizeInches = [18, 14])
               

if(distribution_graph_3D_slices):
    
    i_1 = 2
    i_2 = 0
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mus[i_1],mus[i_2]
    std_1, std_2 = stds[i_1],stds[i_2]
    
    mu = mus[[i_1,i_2]]
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    
    ################ Contour plot of the scatter plot ################
    gl.init_figure()
    std_K= 3
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=30j, ybins=30j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ## Plot the 3D surface
    ax3D = gl.plot_3D(xx, yy, zz, nf = 0,
                      labels = ["slices of the joint distribution","X_1","X_2"], alpha = 0.1)
    
    ## Limits of the plotting !
    xmin,xmax = [np.min(xx.flatten()), np.max(xx.flatten())]
    ymin,ymax = [np.min(yy.flatten()), np.max(yy.flatten())]
    zmin,zmax = [np.min(zz.flatten()), np.max(zz.flatten())]
    
    xmin = xmin - (xmax - xmin)*0.2
    xmax = xmax + (xmax - xmin)*0.2
    ymin = ymin - (ymax - ymin)*0.2
    ymax = ymax + (ymax - ymin)*0.2
    
    xymin, xymax = [np.min([xmin,ymin]),np.max([xmax,ymax])]
    
    # Plot the marginalization of X
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_1, std  = std_1, std_K = std_K)
    y_val = y_val * (1/np.sqrt(2*np.pi*std_2*std_2))
    
#    ax3D.plot(x_grid,y_val,xymax, zdir='y')
#    ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),xymax, zdir='y')
  
    # Plot the marginalization of Y
    # Plot the marginalization of Y
    x_1_list = np.linspace(-3,4,5)
    for x_1 in x_1_list:
        x_grid, y_val = bMA.gaussian1D_points(mean = mu_2, std  = std_2, std_K = std_K) 
        y_val = y_val * (1/np.sqrt(2*np.pi*std_1*std_1))*np.exp(-np.power((x_1 - mu_1),2)/(2*std_1*std_1))
        
        ax3D.plot(x_grid,y_val, xymin, zdir='x')
        ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),x_1, zdir='x')

    # Set the visualization limits !
#    ax3D.set_xlim(xmin, xmax)
#    ax3D.set_ylim(ymin, ymax)
    
    ax3D.set_xlim(xymin, xymax)
    ax3D.set_ylim(xymin, xymax)
    ax3D.set_zlim(zmin, zmax)
    
#    ax1.pcolormesh(xx, yy, zz)
#    ax1.imshow(zz, origin='lower', extent=[-3,3,-3,3], cmap="gray")

    gl.savefig(folder_images +'Gaussian3D_slices.png', 
               dpi = 100, sizeInches = [12, 6])
               

if(distribution_graph_3D_condtional):
    
    i_1 = 2
    i_2 = 0
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mus[i_1],mus[i_2]
    std_1, std_2 = stds[i_1],stds[i_2]
    
    mu = mus[[i_1,i_2]]
    cov = np.cov(np.concatenate((X_1,X_2),axis = 1).T).T
    std_K = 3
    
    ################ Contour plot of the scatter plot ################
    gl.init_figure()
    std_K= 3
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=30j, ybins=30j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ## Plot the 3D surface
    ax3D = gl.plot_3D(xx, yy, zz, nf = 0,
                      labels = ["slices of the joint distribution","X_1","X_2"], alpha = 0.7)
    
    ## Limits of the plotting !
    xmin,xmax = [np.min(xx.flatten()), np.max(xx.flatten())]
    ymin,ymax = [np.min(yy.flatten()), np.max(yy.flatten())]
    zmin,zmax = [np.min(zz.flatten()), np.max(zz.flatten())]
    
    xmin = xmin - (xmax - xmin)*0.2
    xmax = xmax + (xmax - xmin)*0.2
    ymin = ymin - (ymax - ymin)*0.2
    ymax = ymax + (ymax - ymin)*0.2
    
    xymin, xymax = [np.min([xmin,ymin]),np.max([xmax,ymax])]
    
    # Plot the marginalization of X
#    x_grid, y_val = bMA.gaussian1D_points(mean = mu_1, std  = std_1, std_K = std_K)
#    y_val = y_val * (1/np.sqrt(2*np.pi*std_2*std_2))
    
#    ax3D.plot(x_grid,y_val,xymax, zdir='y')
#    ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),xymax, zdir='y')
  
    # Plot the marginalization of Y
    # Plot the marginalization of Y
    x_1_list = np.linspace(-3,4,5)
    for x_1 in x_1_list:
        x_grid, y_val = bMA.gaussian1D_points(mean = mu_2, std  = std_2, std_K = std_K) 
        y_val = y_val # * (1/np.sqrt(2*np.pi*std_1*std_1))*np.exp(-np.power((x_1 - mu_1),2)/(2*std_1*std_1))
        
        ax3D.plot(x_grid,y_val, xymin, zdir='x')
        ax3D.add_collection3d(plt.fill_between(x_grid,y_val, 0, color='k', alpha=0.3),x_1, zdir='x')

    # Set the visualization limits !
#    ax3D.set_xlim(xmin, xmax)
#    ax3D.set_ylim(ymin, ymax)
    
    ax3D.set_xlim(xymin, xymax)
    ax3D.set_ylim(xymin, xymax)
    ax3D.set_zlim(0, np.max(y_val))
    
#    ax1.pcolormesh(xx, yy, zz)
#    ax1.imshow(zz, origin='lower', extent=[-3,3,-3,3], cmap="gray")

    gl.savefig(folder_images +'Gaussian3D_conditional.png', 
               dpi = 100, sizeInches = [12, 6])
               