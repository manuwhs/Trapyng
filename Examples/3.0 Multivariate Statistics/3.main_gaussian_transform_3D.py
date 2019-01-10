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

graph_ellipse = 0
distribution_graph_3D_slices = 1;

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

X = np.concatenate((X),axis = 1).T

Nsim = 1000
x_grid = np.linspace(-6,8,Nsim)
if (graph_ellipse):
    import numpy as np
    import numpy.linalg as linalg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # your ellispsoid and center in matrix form
    S = np.array([[2,0,0],
                  [0,1,0],
                  [0,0,1.5]])
    center = [0,0,0]
    
    theta_xy = 0.2*np.pi
    Rxy = np.array([[np.cos(theta_xy),-np.sin(theta_xy),0],
                   [np.sin(theta_xy),np.cos(theta_xy),0],
                   [0,0,1]])
    theta_xz = 0.3*np.pi
    Rxz = np.array([[np.cos(theta_xz),0,-np.sin(theta_xz)],
                     [0,1,0],
                   [np.sin(theta_xz),0,np.cos(theta_xz)]
                   ])
    
    theta_yz = 0.3*np.pi
    Ryz = np.array([[1,0,0],
                    [0,np.cos(theta_yz),-np.sin(theta_yz)],
                   [0,np.sin(theta_yz),np.cos(theta_yz)]])
    
    R = Rxy.dot(Rxz.dot(Ryz))
    
    
    # find the rotation matrix and radii of the axes
    A = R.dot(S)
    
    U, s, V = linalg.svd(A)
    radii = 4*1.0/np.sqrt(s)
    
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # Create the 
    xx = radii[0] * np.outer(np.cos(u), np.sin(v))
    yy = radii[1] * np.outer(np.sin(u), np.sin(v))
    zz = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    for i in range(len(xx)):
        for j in range(len(xx)):
            [xx[i,j],yy[i,j],zz[i,j]] = A.dot([xx[i,j],yy[i,j],zz[i,j]]) + center
    

#    ax3D = gl.plot_3D(xx, yy, zz, nf = 0,
#                      labels = ["slices of the joint distribution","X_1","X_2"], alpha = 0.2)
#    
    Y = A.dot(X)
#    ax3D = gl.scatter_3D(Y[0,:], Y[1,:], Y[2,:], nf = 0,
#                      labels = ["slices of the joint distribution","X_1","X_2"], alpha = 0.6)
    

    ######## DRAWING THE HYPERPLANE #########
    v = R[0,:]
    
    nx = 1
    grid_x = np.linspace(-nx,nx,20)
    grid_y = np.linspace(-nx,nx,20)

    params = [0, -v[0]/v[2], -v[1]/v[2]]
    zz = bMA.get_plane_Z(grid_x, grid_y, params)
    xx, yy = np.meshgrid(grid_x, grid_y, sparse=True)
    ax3D = gl.plot_3D(xx, yy, zz, nf = 0,
                      labels = ["slices of the joint distribution","X_1","X_2"], alpha = 0.8)

    #### Plotting the vectors
    nx = 5
    v = R[0,:]
    ax3D.plot([-nx*v[0], nx*v[0]],[-nx*v[1], nx*v[1]],[-nx*v[2], nx*v[2]], lw = 3)
    v = R[1,:]
    ax3D.plot([-nx*v[0], nx*v[0]],[-nx*v[1], nx*v[1]],[-nx*v[2], nx*v[2]], lw = 3)
    v = R[2,:]
    ax3D.plot([-nx*v[0], nx*v[0]],[-nx*v[1], nx*v[1]],[-nx*v[2], nx*v[2]], lw = 3)
    
#    ax3D.axis('equal')
#    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    
    ## Limits of the plotting !
    xmin,xmax = [np.min(xx.flatten()), np.max(xx.flatten())]
    ymin,ymax = [np.min(yy.flatten()), np.max(yy.flatten())]
    zmin,zmax = [np.min(zz.flatten()), np.max(zz.flatten())]
    
    xmin = xmin - (xmax - xmin)*0.2
    xmax = xmax + (xmax - xmin)*0.2
    ymin = ymin - (ymax - ymin)*0.2
    ymax = ymax + (ymax - ymin)*0.2
    xymin, xymax = [np.min([xmin,ymin]),np.max([xmax,ymax])]
    
    nval = 10;
    
    ax3D.set_xlim3d(-nval, nval)
    ax3D.set_ylim3d(-nval, nval)
    ax3D.set_zlim3d(-nval, nval)
    plt.show()

    
if(distribution_graph_3D_slices):
    A = 2* np.array([[0.9,2],[2,0.9]])
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
    
    ax3D.set_xlim(xymin, xymax)
    ax3D.set_ylim(xymin, xymax)
    ax3D.set_zlim(zmin, zmax)
    

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
               
