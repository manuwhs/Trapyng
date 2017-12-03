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

distribution_graph_2D = 1
distribution_graph_3D_slices = 0;
distribution_graph_3D_cond = 0
##########################################################################
################# DATA OBTAINING ######################################
##########################################################################
mus = np.array([0,0,0])
stds = np.array([1,1,1])
Nsam = 1000
Nx = 3

X = []
for i in range(Nx):
    X_i = np.random.randn(Nsam,1)*stds[i] + mus[i]
    X.append(X_i)

X = np.concatenate((X),axis = 1)
Nsim = 1000
x_grid = np.linspace(-6,8,Nsim)


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
    
    Xjoint = np.concatenate((X_1,X_2), axis = 1)
    std_1, std_2 = stds[i_1],stds[i_2]
    
    mu = mus[[i_1,i_2]]
    cov = np.cov(Xjoint.T).T
    std_K = 3
    
    ## Do stuff now
    ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    gl.scatter(X_1,X_2, alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["","U1", "U2"])
    
    ax1.axis('equal')
    
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=40j, ybins=40j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ax1.contour(xx, yy, zz, linewidths = 3, linestyles = "solid", alpha = 0.8,
                colors = None)
    
    ######## Transformation !!
    
    A = np.array([[0.9,2],[0.8,0.7]])
    mu = [-1.5,2]
    Yjoint = Xjoint.dot(A) + mu
    
    cov = np.cov(Yjoint.T).T
    
    ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
    gl.scatter(Yjoint[:,0],Yjoint[:,1], alpha = 0.5, ax = ax2, lw = 4, AxesStyle = "Normal",
               labels = ["","X1", "X2"])
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=40j, ybins=40j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ax2.contour(xx, yy, zz, linewidths = 3, linestyles = "solid", alpha = 0.8,
                colors = None)
    
    ax1.set_xlim(-6, 4)
    ax1.set_ylim(-4, 7)
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
    
    gl.savefig(folder_images +'Gaussian_2DX_transform.png', 
               dpi = 100, sizeInches = [18, 9])

    ###############################################################################
    ############################ PLOT ROTATED #####################################
    ###############################################################################
    
    gl.init_figure()
    
    ax1 = gl.subplot2grid((1,3), (0,0), rowspan=1, colspan=1)
    gl.scatter(Yjoint[:,0],Yjoint[:,1], alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["","X1", "X2"])
    
    ax1.axis('equal')
    
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=40j, ybins=40j, mu = mu, cov = cov, 
                      std_K = std_K, x_grid = None)
    ax1.contour(xx, yy, zz, linewidths = 3, linestyles = "solid", alpha = 0.8,
                colors = None)
    
    ######## Transformation !!

    thetas = [0.2*np.pi, 0.6*np.pi]
    for i in range (len(thetas)):
        theta = thetas[i]
        R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta),]])
        Zjoint = (Yjoint - mu).dot(R) + mu
        cov = np.cov(Zjoint.T).T
        
        ax2 = gl.subplot2grid((1,3), (0,1+i), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
        gl.scatter(Zjoint[:,0],Zjoint[:,1], alpha = 0.5, ax = ax2, lw = 4, AxesStyle = "Normal",
                   labels = ["theta: %f pi"%(theta/(np.pi)),"Y1", "Y2"])
        
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)
    
        xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=40j, ybins=40j, mu = mu, cov = cov, 
                          std_K = std_K, x_grid = None)
        
        ax2.contour(xx, yy, zz, linewidths = 3, linestyles = "solid", alpha = 0.8,
                    colors = None)
        
        ax1.set_xlim(-6, 4)
        ax1.set_ylim(-4, 7)
        
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)
    
    gl.savefig(folder_images +'Gaussian_2DX_transform_rot.png', 
               dpi = 100, sizeInches = [18, 9])
    
if(distribution_graph_3D_slices):
    A = np.array([[0.9,2],[2,0.9]])
    mu = [-1.5,2]
    
    i_1 = 0
    i_2 = 1
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mu[i_1],mu[i_2]
    
    Xjoint = np.concatenate((X_1,X_2), axis = 1)

    
    std_K = 3
    
    Yjoint = Xjoint.dot(A) + mu
    cov = np.cov(Yjoint.T).T
    std_1, std_2 = np.sqrt(cov[i_1,i_1]),np.sqrt(cov[i_2,i_2])
    
    ################ Contour plot of the scatter plot ################
    gl.init_figure()
    std_K= 3
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=60j, ybins=60j, mu = mu, cov = cov, 
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
    
    # Plot the marginalization of Y
    x_1_list = np.linspace(-7,2,5)
    
    i1 = 0;
    i2= 1;
    for x_1 in x_1_list:
        # Compute the conditional mu_2 and std_2
        L = (cov[i2,i1]/cov[i1,i1])
        mu_2_cond = mu_2 + L*(x_1 - mu_1)
        sigma_2_cond = np.sqrt(cov[i2,i2] - L*cov[i1,i2])
        
        x_grid, y_val = bMA.gaussian1D_points(mean = mu_2_cond, std  = sigma_2_cond, std_K = std_K) 
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

    gl.savefig(folder_images +'Gaussian3D_slices_dependent.png', 
               dpi = 100, sizeInches = [12, 6])
               

if(distribution_graph_3D_slices):
    A = np.array([[0.9,2],[2,0.9]])
    mu = [-1.5,2]
    
    i_1 = 0
    i_2 = 1
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mu[i_1],mu[i_2]
    
    Xjoint = np.concatenate((X_1,X_2), axis = 1)

    
    std_K = 3
    
    Yjoint = Xjoint.dot(A) + mu
    cov = np.cov(Yjoint.T).T
    std_1, std_2 = np.sqrt(cov[i_1,i_1]),np.sqrt(cov[i_2,i_2])
    
    ################ Contour plot of the scatter plot ################
    gl.init_figure()
    std_K= 3
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=60j, ybins=60j, mu = mu, cov = cov, 
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
    
    # Plot the marginalization of Y
    x_1_list = np.linspace(-7,2,5)
    
    i1 = 0;
    i2= 1;
    for x_1 in x_1_list:
        # Compute the conditional mu_2 and std_2
        L = (cov[i2,i1]/cov[i1,i1])
        mu_2_cond = mu_2 + L*(x_1 - mu_1)
        sigma_2_cond = np.sqrt(cov[i2,i2] - L*cov[i1,i2])
        
        x_grid, y_val = bMA.gaussian1D_points(mean = mu_2_cond, std  = sigma_2_cond, std_K = std_K) 
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

    gl.savefig(folder_images +'Gaussian3D_slices_dependent.png', 
               dpi = 100, sizeInches = [12, 6])
               

if(distribution_graph_3D_cond):
    A = np.array([[0.9,2],[2,0.8]])
    mu = [-1.5,2]
    
    i_1 = 0
    i_2 = 1
    X_1,X_2 = X[:,[i_1]], X[:,[i_2]]
    mu_1, mu_2  = mu[i_1],mu[i_2]
    
    Xjoint = np.concatenate((X_1,X_2), axis = 1)

    
    std_K = 3
    
    Yjoint = Xjoint.dot(A) + mu
    cov = np.cov(Yjoint.T).T
    std_1, std_2 = np.sqrt(cov[i_1,i_1]),np.sqrt(cov[i_2,i_2])
    
    ################ Contour plot of the scatter plot ################
    gl.init_figure()
    std_K= 3
    xx, yy, zz = bMA.get_gaussian2D_pdf( xbins=60j, ybins=60j, mu = mu, cov = cov, 
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
    
    # Plot the marginalization of Y
    x_1_list = np.linspace(-7,2,5)
    
    i1 = 0;
    i2= 1;
    for x_1 in x_1_list:
        # Compute the conditional mu_2 and std_2
        L = (cov[i2,i1]/cov[i1,i1])
        mu_2_cond = mu_2 + L*(x_1 - mu_1)
        sigma_2_cond = np.sqrt(cov[i2,i2] - L*cov[i1,i2])
        
        x_grid, y_val = bMA.gaussian1D_points(mean = mu_2_cond, std  = sigma_2_cond, std_K = std_K) 
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

    gl.savefig(folder_images +'Gaussian3D_cond_dependent.png', 
               dpi = 100, sizeInches = [12, 6])
               
    