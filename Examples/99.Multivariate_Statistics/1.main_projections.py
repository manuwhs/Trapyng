"""
We generate random Gaussian data and project into another base.
It is a didactic shit
"""

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
import scipy

plt.close("all")

folder_images = "../pics/Trapying/MultivariateStat/"
##############################################
########## FLAGS ############################
distribution_graph = 1
ellipse_graph = 1


############################################################
################# Create Data DATA ###############################
###########################################################

mus = np.array([0,0])
stds = np.array([1,1])
Nsam = 10
Nx = 2

X = []
for i in range(Nx):
    X_i = np.random.randn(Nsam,1)*stds[i] + mus[i]
    X.append(X_i)

X = np.concatenate((X),axis = 1).T
Nsim = 1000
x_grid = np.linspace(-6,8,Nsim)

### Transform the data:
S = np.diag([2,1])
theta = (35.0/360) * (2*np.pi)
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

mu_Y = np.array([[0],[0]])

Y = R.dot(S.dot(X)) + mu_Y
SigmaY = R.dot(S.dot(S.T.dot(R.T)))

# We dont really want to compute it at the end since too little samples is inacurate
cov = np.cov(Y)
mu = np.mean(Y,axis = 1)
mu.resize(2,1)

## Projection to new basis
theta2 = -(15.0/360) * (2*np.pi)
R = np.array([[np.cos(theta2),-np.sin(theta2)],[np.sin(theta2),np.cos(theta2)]])
Z = R.dot(Y - mu_Y) + mu_Y
SigmaZ = R.dot(SigmaY.dot(R.T))
mu_Z = mu_Y

## TODO: AS we can see, the angle is actually positive, but when proejcting that is what you lose.
############################################################
################# PLOT DATA ###############################
############################################################


if(distribution_graph):
    # Get the histogram and gaussian estimations !
    ## Scatter plot of the points 
    
    gl.init_figure()
    ax1 = gl.subplot2grid((4,4), (1,0), rowspan=3, colspan=3)
    gl.scatter(Y[0,:],Y[1,:], alpha = 0.5, ax = ax1, lw = 4, AxesStyle = "Normal",
               labels = ["","U1","U2"],
               legend = ["%i points"%Nsam])
    
    # Plot the projection vectors
    n = 6;
    gl.plot([-n*R[0,0],n*R[0,0]],[- n*R[0,1],n*R[0,1]],color = "y", lw = 3);
    gl.plot([-n*R[1,0],n*R[1,0]],[-n*R[1,1],n*R[1,1]],color = "r", lw = 3);
    
    ## Plot the projections !!
#    V1_pr= []
#    V2_pr= []
#    
#    for i in range(Y.shape[1]):
#        V1_pr.append([Y[:,i],Y[:,i]- R[0,:].dot(Y[:,i]) * R[0,:]])
#        V2_pr.append([Y[:,i],Y[:,i] - R[1,:].dot(Y[:,i]) * R[1,:]])
#        
#        gl.plot([V1_pr[i][0][0],V1_pr[i][1][0]],[V1_pr[i][0][1],V1_pr[i][1][1]], color = "y")
#        gl.plot([V2_pr[i][0][0],V2_pr[i][1][0]],[V2_pr[i][0][1],V2_pr[i][1][1]], color = "r")
#        gl.plot(V2_pr[i][0],V2_pr[i][1], color = "y")
#    gl.plot([mu[0],n*R[0,0] + mu[0]],[mu[1],n*R[0,1]+ mu[1]],color = "y");
#    gl.plot([mu[0],n*R[1,0] + mu[0]],[mu[1],n*R[1,1]+ mu[1]],color = "y");
    
    ## X distribution
    ax2 = gl.subplot2grid((4,4), (0,0), rowspan=1, colspan=3, sharex = ax1)
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_Y[0], std = SigmaY[0,0], std_K = 3)
    gl.plot(x_grid, y_val, color = "k",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mu_Y[0], SigmaY[0,0])])
    
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_Z[0], std = SigmaZ[0,0], std_K = 3)
    gl.plot(x_grid, y_val, color = "g",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mu_Z[0], SigmaZ[0,0])])
    
    # Y distribution
    ax3 = gl.subplot2grid((4,4), (1,3), rowspan=3, colspan=1,sharey = ax1,) 
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_Y[1], std = SigmaY[1,1], std_K = 3)
    gl.plot(y_val, x_grid, color = "k",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mu_Y[1], SigmaY[1,1])])
    x_grid, y_val = bMA.gaussian1D_points(mean = mu_Z[1], std = SigmaZ[1,1], std_K = 3)
    gl.plot( y_val,x_grid , color = "g",
            labels = ["","",""], legend = ["M: %.2e, std: %.2e"%(mu_Z[1], SigmaZ[1,1])])
    ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    if(ellipse_graph):
        ################# Draw the error ellipse  #################
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu_Y, Sigma = SigmaY, Chi2val = 2.4477)
#        mean,vecs = bMA.get_gaussian_mean_and_vects(Y.T)
        vecs,vals = bMA.get_eigenVectorsAndValues(Sigma = SigmaY)
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2,
                 legend = ["Corr: .2f"],AxesStyle = "Normal2")
        
        gl.plot([mean[0] - vecs[0,0]*w, mean[0] + vecs[0,0]*w], 
                [mean[1] - vecs[0,1]*w, mean[1] + vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
        gl.plot([mean[0] - vecs[1,0]*h, mean[0] + vecs[1,0]*h], 
                [mean[1] - vecs[1,1]*h, mean[1] + vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
        

        ax1.axis('equal')
        gl.set_zoom(ax = ax1, X =r_ellipse[:,0], Y = r_ellipse[:,1],
                    ylimPad = [0.2,0.2],xlimPad = [0.2,0.2])

        
    gl.savefig(folder_images +'RotatedProjection.png', 
               dpi = 100, sizeInches = [14, 7])
    
    

############################################################
################# PLOT DATA ###############################
###########################################################

## Now we are gonna plot the projections and the final thing

    gl.set_subplots(1,3)

    ### First projections 
    ax1 = gl.scatter(Y[0,:],Y[1,:], alpha = 0.5, lw = 4, AxesStyle = "Normal",
               labels = ["","U1","U2"],
               legend = ["%i points"%Nsam], nf = 1)
    
    # Plot the projection vectors
    n = 6;
    gl.plot([-n*R[0,0],n*R[0,0]],[- n*R[0,1],n*R[0,1]],color = "y", lw = 3);
    gl.plot([-n*R[1,0],n*R[1,0]],[-n*R[1,1],n*R[1,1]],color = "r", lw = 3);
    
    ## Plot the projections !!
    V1_pr= []
    for i in range(Y.shape[1]):
        V1_pr.append([Y[:,i],Y[:,i]- R[0,:].dot(Y[:,i]) * R[0,:]])
        gl.plot([V1_pr[i][0][0],V1_pr[i][1][0]],[V1_pr[i][0][1],V1_pr[i][1][1]], color = "y")

    ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    if(ellipse_graph):
        ################# Draw the error ellipse  #################
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu_Y, Sigma = SigmaY, Chi2val = 2.4477)
#        mean,vecs = bMA.get_gaussian_mean_and_vects(Y.T)
        vecs,vals = bMA.get_eigenVectorsAndValues(Sigma = SigmaY)
        
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2,
                 legend = ["Corr: .2f"],AxesStyle = "Normal2")
        
        gl.plot([mean[0] - vecs[0,0]*w, mean[0] + vecs[0,0]*w], 
                [mean[1] - vecs[0,1]*w, mean[1] + vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
        gl.plot([mean[0] - vecs[1,0]*h, mean[0] + vecs[1,0]*h], 
                [mean[1] - vecs[1,1]*h, mean[1] + vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
    

    ### Second projections 
    ax1 = gl.scatter(Y[0,:],Y[1,:], alpha = 0.5, lw = 4, AxesStyle = "Normal",
               labels = ["","U1","U2"],
               legend = ["%i points"%Nsam], nf = 1, sharex = ax1, sharey=ax1)
    
    # Plot the projection vectors
    n = 6;
    gl.plot([-n*R[0,0],n*R[0,0]],[- n*R[0,1],n*R[0,1]],color = "y", lw = 3);
    gl.plot([-n*R[1,0],n*R[1,0]],[-n*R[1,1],n*R[1,1]],color = "r", lw = 3);
    
    ## Plot the projections !!
    V2_pr= []
    
    for i in range(Y.shape[1]):
        V2_pr.append([Y[:,i],Y[:,i] - R[1,:].dot(Y[:,i]) * R[1,:]])
        gl.plot([V2_pr[i][0][0],V2_pr[i][1][0]],[V2_pr[i][0][1],V2_pr[i][1][1]], color = "r")

    ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    if(ellipse_graph):
        ################# Draw the error ellipse  #################
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu_Y, Sigma = SigmaY, Chi2val = 2.4477)
#        mean,vecs = bMA.get_gaussian_mean_and_vects(Y.T)
        vecs,vals = bMA.get_eigenVectorsAndValues(Sigma = SigmaY)
        
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2,
                 legend = ["Corr: .2f"],AxesStyle = "Normal2")
        
        gl.plot([mean[0] - vecs[0,0]*w, mean[0] + vecs[0,0]*w], 
                [mean[1] - vecs[0,1]*w, mean[1] + vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
        gl.plot([mean[0] - vecs[1,0]*h, mean[0] + vecs[1,0]*h], 
                [mean[1] - vecs[1,1]*h, mean[1] + vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
        
    ### Last projections projections 
    
    ax1 = gl.scatter(Z[0,:],Z[1,:], alpha = 0.5, lw = 4, AxesStyle = "Normal",
               labels = ["","U1","U2"],
               legend = ["%i points"%Nsam], nf = 1, sharex = ax1, sharey=ax1)
    n = 6;
    gl.plot([-n,n],[0,0],color = "y", lw = 3);
    gl.plot([0,0],[-n,n],color = "r", lw = 3);
    
    ax1.axis('equal')
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.01, hspace=0.01)

    if(ellipse_graph):
        ################# Draw the error ellipse  #################
        mean,w,h,theta = bMA.get_gaussian_ellipse_params( mu = mu_Z, Sigma = SigmaZ, Chi2val = 2.4477)
#        mean,vecs = bMA.get_gaussian_mean_and_vects(Y.T)
        vecs,vals = bMA.get_eigenVectorsAndValues(Sigma = SigmaY)
        
        r_ellipse = bMA.get_ellipse_points(mean,w,h,theta)
        gl.plot(r_ellipse[:,0], r_ellipse[:,1], ax = ax1, ls = "--",color = "k", lw = 2,
                 legend = ["Corr: .2f"],AxesStyle = "Normal2")
        
        gl.plot([mean[0] - vecs[0,0]*w, mean[0] + vecs[0,0]*w], 
                [mean[1] - vecs[0,1]*w, mean[1] + vecs[0,1]*w], ax = ax1, ls = "--",color = "k")
        gl.plot([mean[0] - vecs[1,0]*h, mean[0] + vecs[1,0]*h], 
                [mean[1] - vecs[1,1]*h, mean[1] + vecs[1,1]*h], ax = ax1, ls = "--",color = "k")
    

    gl.savefig(folder_images +'ProjectionsDone.png', 
               dpi = 100, sizeInches = [18, 7])
    