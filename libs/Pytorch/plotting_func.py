"""
This is a nice small library to plot things related to the training of the algorithms !!
"""

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import utilities_lib as ul
import numpy as np
import pandas as pd
import torch
# Own graphical library
from graph_lib import gl
# Data Structures Data

import video_utils as vul

from matplotlib import cm as cm
import basicMathlib as bMA

import Variational_inferences_lib as Vil

"""
############### MAIN FUNCTIONS THAT CALL OTHERS TO CREATE THE FINAL IMAGE AND SAVE TO DISK ##########
"""
def create_Bayesian_analysis_charts_simplified(model, train_dataset, validation_dataset,
                                    tr_loss, val_loss, KL_loss,
                                    folder_images,
                                    epoch_i = None):

    # Configurations of the plots
    alpha_points = 0.2 
    color_points_train = "dark navy blue"
    color_points_val = "amber"
    color_train_loss = "cobalt blue"
    color_val_loss = "blood"
    color_truth = "k"
    color_mean = "b"
    color_most_likey = "y"

    ################################ Divide in plots ##############################
    gl.init_figure();
    ax1 = gl.subplot2grid((6,3), (0,0), rowspan=3, colspan=1)
    ax2 = gl.subplot2grid((6,3), (3,0), rowspan=3, colspan=1, sharex = ax1, sharey = ax1)
    
    ax3 = gl.subplot2grid((6,3), (0,1), rowspan=2, colspan=1)
    ax4 = gl.subplot2grid((6,3), (2,1), rowspan=2, colspan=1, sharex = ax3)
    ax5 = gl.subplot2grid((6,3), (4,1), rowspan=2, colspan=1, sharex = ax3)
    
    ax6 = gl.subplot2grid((6,3), (0,2), rowspan=3, colspan=1)
    ax7 = gl.subplot2grid((6,3), (3,2), rowspan=3, colspan=1, sharex = ax6)
    
    
   ####### ax1, ax2: Get confusion matrices ##########

    labels_classes, confusion = model.get_confusion_matrix(train_dataset)
    plot_confusion_matrix(confusion,labels_classes, ax1 )
    labels_classes, confusion = model.get_confusion_matrix(validation_dataset)
    plot_confusion_matrix(confusion,labels_classes, ax2 )
        
   ############## ax3 ax4 ax5: Loss Evolution !! ######################
    ## ax3: Evolutoin of the data loss
    gl.plot([], tr_loss, ax = ax3, lw = 3, labels = ["Losses", "","Data loss (MSE)"], legend = ["train"],
            color = color_train_loss)
    gl.plot([], val_loss,ax = ax3, lw = 3, legend = ["validation"],
            color = color_val_loss,  AxesStyle = "Normal - No xaxis")
    
    ## ax4: The evolution of the KL loss
    gl.plot([], KL_loss, ax = ax4, lw = 3, labels = ["", "","KL loss"], legend = ["Bayesian Weights"],
            AxesStyle = "Normal - No xaxis", color = "k")

    ## ax5: Evolutoin of the total loss
    gl.plot([], tr_loss, ax = ax5, lw = 3, labels = ["", "epoch","Total Loss (Bayes)"], legend = ["train"],
            color = color_train_loss)
    gl.plot([], val_loss,ax = ax5, lw = 3, legend = ["validation"], color = color_val_loss)
           
    ############## ax6 ax7: Variational Weights !! ######################
    create_plot_variational_weights(model,ax6,ax7)

    gl.set_zoom (ax = ax6, ylim = [-0.1,10])
    gl.set_zoom (ax = ax7, xlim = [-2.5, 2.5], ylim = [-0.1,0.5])
    
    # Set final properties and save figure
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 10, xticks = 12, yticks = 12)


    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    
    if (type(epoch_i) == type(None)):
        gl.savefig(folder_images +'Training_Example_Data_Bayesian.png', 
                   dpi = 100, sizeInches = [20, 10])
    else:
        gl.savefig(folder_images +'%i.png'%epoch_i, 
                   dpi = 100, sizeInches = [20, 10], close = True, bbox_inches = "tight")
    
    

def create_Bayesian_analysis_charts(model,
                                    X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                    tr_loss, val_loss, KL_loss,final_loss_tr,final_loss_val,
                                    xgrid_real_func, ygrid_real_func,
                                    folder_images,
                                    epoch_i = None):

    # Configurations of the plots
   
    alpha_points = 0.2 
    color_points_train = "dark navy blue"
    color_points_val = "amber"
    color_train_loss = "cobalt blue"
    color_val_loss = "blood"
    color_truth = "k"
    color_mean = "b"
    color_most_likey = "y"

    ############################# Data computation #######################
    if(type(X_data_tr) == type([])):
        pass
    else:
        if (X_data_tr.shape[1] == 1): # Regression Example 
            x_grid, all_y_grid,most_likely_ygrid = compute_regression_1D_data( model,X_data_tr,X_data_val, Nsamples = 100)
        elif(X_data_tr.shape[1] == 2):  # Classification Example 
            xx,yy , all_y_grid,most_likely_ygrid = compute_classification_2D_data( model,X_data_tr,X_data_val, Nsamples = 100)
        else:        # RNN
            x_grid, all_y_grid,most_likely_ygrid = compute_RNN_1D_data( model,X_data_tr,X_data_val, Nsamples = 100)
        
    ################################ Divide in plots ##############################
    gl.init_figure();
    ax1 = gl.subplot2grid((6,3), (0,0), rowspan=3, colspan=1)
    ax2 = gl.subplot2grid((6,3), (3,0), rowspan=3, colspan=1, sharex = ax1, sharey = ax1)
    
    ax3 = gl.subplot2grid((6,3), (0,1), rowspan=2, colspan=1)
    ax4 = gl.subplot2grid((6,3), (2,1), rowspan=2, colspan=1, sharex = ax3)
    ax5 = gl.subplot2grid((6,3), (4,1), rowspan=2, colspan=1, sharex = ax3)
    
    ax6 = gl.subplot2grid((6,3), (0,2), rowspan=3, colspan=1)
    ax7 = gl.subplot2grid((6,3), (3,2), rowspan=3, colspan=1, sharex = ax6)
    
    if(type(X_data_tr) == type([])):
        Xtrain = [torch.tensor(X_data_tr[i],device=model.cf_a.device, dtype=model.cf_a.dtype) for i in range(len(X_data_tr))]
        Ytrain = torch.tensor(Y_data_tr,device=model.cf_a.device, dtype=torch.int64)
        
        Xval = [torch.tensor(X_data_val[i],device=model.cf_a.device, dtype=model.cf_a.dtype) for i in range(len(X_data_val))]
        Yval = torch.tensor(Y_data_val,device=model.cf_a.device, dtype=torch.int64)

        confusion = model.get_confusion_matrix(Xtrain, Ytrain)
        plot_confusion_matrix(confusion,model.languages, ax1 )
        confusion = model.get_confusion_matrix(Xval, Yval)
        plot_confusion_matrix(confusion,model.languages, ax2 )

    else:
        if (X_data_tr.shape[1] == 1): # Regression Example 
            plot_data_regression_1d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                              x_grid,all_y_grid, most_likely_ygrid,
                                              alpha_points, color_points_train, color_points_val, color_most_likey,color_mean,color_truth,
                                              ax1,ax2)
        elif(X_data_tr.shape[1] == 2): # Classification Example 
            plot_data_classification_2d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                               xx,yy,all_y_grid, most_likely_ygrid,
                                              alpha_points, color_points_train, color_points_val, color_most_likey,color_mean, color_truth,
                                              ax1,ax2)
        else:       # RNN example
            plot_data_RNN_1d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                              x_grid,all_y_grid, most_likely_ygrid,
                                              alpha_points, color_points_train, color_points_val, color_most_likey,color_mean,color_truth,
                                              ax1,ax2)
 
#    gl.fill_between (x_grid, [mean_samples_grid + 2*std_samples_grid, mean_samples_grid - 2*std_samples_grid]
#                              , ax  = ax2, alpha = 0.10, color = "b", legend = ["Mean realizaions"])
    ## ax2: The uncertainty of the prediction !!
#    gl.plot (x_grid, std_samples_grid, ax = ax2, labels = ["Std (%i)"%(Nsamples),"X","f(X)"], legend = [" std predictions"], fill = 1, alpha = 0.3)
    
   ############## ax3 ax4 ax5: Loss Evolution !! ######################
    ## ax3: Evolutoin of the data loss
    gl.plot([], tr_loss, ax = ax3, lw = 3, labels = ["Losses", "","Data loss"], legend = ["train"],
            color = color_train_loss)
    gl.plot([], val_loss,ax = ax3, lw = 3, legend = ["validation"],
            color = color_val_loss,  AxesStyle = "Normal - No xaxis")
    
    ## ax4: The evolution of the KL loss
    gl.plot([], KL_loss, ax = ax4, lw = 3, labels = ["", "","KL loss"], legend = ["Bayesian Weights"],
            AxesStyle = "Normal - No xaxis", color = "k")

    ## ax5: Evolutoin of the total loss
    gl.plot([], final_loss_tr, ax = ax5, lw = 3, labels = ["", "epoch","Total Loss (Bayes)"], legend = ["train"],
            color = color_train_loss)
    gl.plot([], final_loss_val,ax = ax5, lw = 3, legend = ["validation"], color = color_val_loss)
           
    ############## ax6 ax7: Variational Weights !! ######################
    create_plot_variational_weights(model,ax6,ax7)
    ## Plot in chart 7 the acceptable mu = 2sigma  -> sigma = |mu|/2sigma 
    mu_grid = np.linspace(-3,3,100)
    y_grid = np.abs(mu_grid)/2
    
    gl.fill_between(mu_grid, 10*np.ones(mu_grid.size), y_grid,
                    alpha = 0.2, color = "r", ax = ax7, legend = ["95% non-significant"])
    
    gl.set_zoom (ax = ax6, ylim = [-0.1,10])
    gl.set_zoom (ax = ax7, xlim = [-2.5, 2.5], ylim = [-0.05, np.exp(model.cf_a.input_layer_prior["log_sigma2"])*(1 + 0.15)])
    
#    gl.set_zoom (ax = ax7, xlim = [-2.5, 2.5], ylim = [-0.1,2])
    
    # Set final properties and save figure
    gl.set_fontSizes(ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 10, xticks = 12, yticks = 12)


    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    
    if (type(epoch_i) == type(None)):
        gl.savefig(folder_images +"../"+'Final_values_regression_1D_' +str(model.cf_a.eta_KL) +'.png', 
                   dpi = 100, sizeInches = [20, 10])
    else:
        gl.savefig(folder_images +'%i.png'%epoch_i, 
                   dpi = 100, sizeInches = [20, 10], close = True, bbox_inches = "tight")
    
    
def create_image_weights_epoch(model, video_fotograms_folder2, epoch_i):
    """
    Creates the image of the training and validation accuracy
    """
    N_Bayesian_layers = len(model.VBmodels)    
    N_Normal_layers = len(model.LinearModels)
    
    # Compute the number of squares we will need:
    # 1 x linear layers, 2 x LSTMS
    
    gl.init_figure();
    cmap = cm.get_cmap('coolwarm', 30)
    
    all_axes = []
    for i in range(N_Bayesian_layers):
        layer = model.VBmodels[i]
        
#        if (layer.type_layer == "linear"):
        if ("linear" in type(layer).__name__.lower()):
            ax = gl.subplot2grid((1,N_Bayesian_layers + N_Normal_layers), (0,i), rowspan=1, colspan=1)
            weights = layer.weight.detach().cpu().numpy()
            biases = layer.bias.detach().cpu().numpy().reshape(-1,1)
            neurons = np.concatenate((weights, biases), axis = 1)
            cax = ax.imshow(neurons, interpolation="nearest", cmap=cmap, vmin=-2, vmax=2)
        
            all_axes.append(ax)
        else:
            ax = gl.subplot2grid((1,N_Bayesian_layers + N_Normal_layers), (0,i), rowspan=1, colspan=1)
            weights_ih = layer.weight_ih.detach().cpu().numpy()
            biases_ih = layer.bias_ih.detach().cpu().numpy().reshape(-1,1)
            weights_hh = layer.weight_hh.detach().cpu().numpy()
            biases_hh = layer.bias_hh.detach().cpu().numpy().reshape(-1,1)
            
            weights = np.concatenate((weights_ih,weights_hh),axis = 1)
            biases = np.concatenate((biases_ih,biases_hh),axis = 1)
            neurons = np.concatenate((weights, biases), axis = 1)
            cax = ax.imshow(neurons, interpolation="nearest", cmap=cmap, vmin=-2, vmax=2)
            all_axes.append(ax)
            
            
    for i in range(N_Normal_layers):
        layer = model.LinearModels[i]
        if ("linear" in type(layer).__name__.lower()):
            ax = gl.subplot2grid((1,N_Bayesian_layers + N_Normal_layers), (0,N_Bayesian_layers +i), rowspan=1, colspan=1)
            weights = layer.weight.detach().cpu().numpy()
            biases = layer.bias.detach().cpu().numpy().reshape(-1,1)
            neurons = np.concatenate((weights, biases), axis = 1)
            cax = ax.imshow(neurons, interpolation="nearest", cmap=cmap, vmin=-2, vmax=2)
            all_axes.append(ax)
        else:
            ax = gl.subplot2grid((1,N_Bayesian_layers + N_Normal_layers), (0,N_Bayesian_layers +i), rowspan=1, colspan=1)
            weights_ih = layer.weight_ih.detach().cpu().numpy()
            biases_ih = layer.bias_ih.detach().cpu().numpy().reshape(-1,1)
            weights_hh = layer.weight_hh.detach().cpu().numpy()
            biases_hh = layer.bias_hh.detach().cpu().numpy().reshape(-1,1)
            
            weights = np.concatenate((weights_ih,weights_hh),axis = 1)
            biases = np.concatenate((biases_ih,biases_hh),axis = 1)
            neurons = np.concatenate((weights, biases), axis = 1)
            cax = ax.imshow(neurons, interpolation="nearest", cmap=cmap, vmin=-2, vmax=2)
            all_axes.append(ax)
            
#    plt.xticks(range(data_df_train.shape[1]), data_df_train.columns, rotation='vertical')
#    plt.yticks(range(data_df_train.shape[1]), data_df_train.columns, rotation='horizontal')
    plt.colorbar(cax)
#    plt.colorbar(cax2)
#        ax1.set_xticks(data_df_train.columns) # , rotation='vertical'
#    ax1.grid(True)
    plt.title('Weights ')

    
#    labels=[str(x) for x in range(Nshow )]
#    ax1.set_xticklabels(labels,fontsize=20)
#    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.show()

    
    gl.set_fontSizes(ax = [all_axes], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 20, xticks = 12, yticks = 12)
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.30)
    
    gl.savefig(video_fotograms_folder2 +'%i.png'%epoch_i, 
               dpi = 100, sizeInches = [14, 10], close = True, bbox_inches = None)
        
"""
 #################### UTILITIES ##############################
"""

def create_video_from_images(video_fotograms_folder, output_file = "out.avi", fps = 2):
    images_path = ul.get_allPaths(video_fotograms_folder)
    images_path = sorted(images_path, key = ul.cmp_to_key(ul.filenames_comp))
#    print(images_path)
    vul.create_video(images_path, output_file = output_file, fps = fps)


def make_meshgrid(x, y, h=.02):
    """
    Make a 2D mesh for the 2D classification algorithm
    """
    
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

"""
####################### REGRESSION EXAMPLE SPECIFIC ###########################3
"""
def compute_regression_1D_data( model,X_data_tr,X_data_val, Nsamples = 100):
    """
    This function computes the outputs of the Regression model for the 1D example
    """
    device = model.cf_a.device
    dtype = model.cf_a.dtype
    model.set_posterior_mean(False)
    
    ################ Obtain the data: ##################
    x_grid = np.linspace(np.min([X_data_tr]) -1, np.max([X_data_val]) +1, 1000)
    all_y_grid = []  ## Used to compute the variance of the prediction !!
    for i in range(Nsamples):
#        print (y_grid[0])
        y_grid =  model.predict(torch.tensor(x_grid.reshape(-1,1),device=device, dtype=dtype)).detach().cpu().numpy()
        all_y_grid.append(y_grid)
        
    all_y_grid = np.concatenate(all_y_grid, axis = 1)
    model.set_posterior_mean(True)
    most_likely_ygrid =  model.predict(torch.tensor(x_grid.reshape(-1,1),device=device, dtype=dtype)).detach().cpu().numpy()
    
    return x_grid, all_y_grid,most_likely_ygrid

def plot_data_regression_1d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                  x_grid,all_y_grid, most_likely_ygrid,
                                  alpha_points, color_points_train, color_points_val, color_most_likey,color_mean, color_truth,
                                  ax1,ax2):
    """
    This function plots the outputs of the Regression model for the 1D example
    """
    
    ## Compute mean and std of regression
    std_samples_grid = np.std(all_y_grid, axis = 1)
    mean_samples_grid = np.mean(all_y_grid, axis = 1)
    
    ############## ax1: Data + Mostlikely + Real + Mean !! ########################
    if(type(ax1) != type(None)):
        gl.scatter(X_data_tr, Y_data_tr, ax = ax1, lw = 3,  #legend = ["tr points"], 
                   labels = ["Data and predictions", "","Y"], alpha = alpha_points, color = color_points_train)
        gl.scatter(X_data_val, Y_data_val, ax = ax1, lw = 3, #legend = ["val points"], 
                   alpha = alpha_points, color = color_points_val)
        
        gl.plot (xgrid_real_func, ygrid_real_func, ax  = ax1, alpha = 0.90, color = color_truth, legend = ["Truth"])
        gl.plot (x_grid, most_likely_ygrid, ax  = ax1, alpha = 0.90, color = color_most_likey, legend = ["Most likely"])
        gl.plot (x_grid, mean_samples_grid, ax  = ax1, alpha = 0.90, color = color_mean, legend = ["Posterior mean"],
                 AxesStyle = "Normal - No xaxis")
    
    ############## ax2: Data + Realizations of the function !! ######################
    if(type(ax2) != type(None)):
        gl.scatter(X_data_tr, Y_data_tr, ax = ax2, lw = 3,  # legend = ["tr points"], 
                   labels = ["", "X","Y"], alpha = alpha_points, color = color_points_train)
        gl.scatter(X_data_val, Y_data_val, ax = ax2, lw = 3, # legend = ["val points"], 
                   alpha = alpha_points, color = color_points_val)
            
        gl.plot (x_grid, all_y_grid, ax  = ax2, alpha = 0.15, color = "k")
        gl.plot (x_grid, mean_samples_grid, ax  = ax2, alpha = 0.90, color = "b", legend = ["Mean realization"])
        
    gl.set_zoom(xlimPad = [0.2,0.2], ylimPad = [0.2,0.2], ax = ax2, X = X_data_tr, Y = Y_data_tr)


"""
####################### CLASSIFICATION EXAMPLE SPECIFIC ###########################3
"""
def compute_classification_2D_data( model,X_data_tr,X_data_val, Nsamples = 100):
    """
    This function computes the outputs of the Classification model for the 2D example
    """
    
    device = model.cf_a.device
    dtype = model.cf_a.dtype
    model.set_posterior_mean(False)
    
    ################ Obtain the data: ##################
    xx,yy = make_meshgrid(X_data_tr[:,0], X_data_tr[:,1])
    x_grid = np.c_[xx.ravel(), yy.ravel()]
    all_y_grid = []  ## Used to compute the variance of the prediction !!
    for i in range(Nsamples):
#        print (y_grid[0])
        y_grid =  model.predict(torch.tensor(x_grid,device=device, dtype=dtype)).detach().cpu().numpy()
        all_y_grid.append(y_grid)
        
#    all_y_grid = np.concatenate(all_y_grid, axis = 1)
    model.set_posterior_mean(True)
    most_likely_ygrid =  model.predict(torch.tensor(x_grid,device=device, dtype=dtype)).detach().cpu().numpy()
    
    return xx,yy , all_y_grid,most_likely_ygrid

def plot_data_classification_2d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                  xx,yy,all_y_grid, most_likely_ygrid,
                                  alpha_points, color_points_train, color_points_val, color_most_likey,color_mean, color_truth,
                                  ax1,ax2):
    """
    This function plots the outputs of the Classification model for the 2D example
    """
    
    alpha_points = 1
    ## Compute mean and std of regression
    std_samples_grid = np.std(all_y_grid, axis = 1)
    mean_samples_grid = np.mean(all_y_grid, axis = 1)
    
    ############## ax1: Data + Mostlikely + Real + Mean !! ########################
    
    classes = np.unique(Y_data_tr).flatten();
    colors = ["r","g","b"]
    
    for i in range(classes.size):
        X_data_tr_class = X_data_tr[np.where(Y_data_tr == classes[i])[0],:]
        X_data_val_class = X_data_val[np.where(Y_data_val == classes[i])[0],:]
#        print (X_data_tr_class.shape)
#        print (classes)
#        print (X_data_tr)
        if ((X_data_tr_class.size > 0) and (X_data_val_class.size > 0)):
            gl.scatter(X_data_tr_class[:,0].flatten().tolist(), X_data_tr_class[:,1].flatten().tolist(), ax = ax1, lw = 3,  #legend = ["tr points"], 
                       labels = ["Data and predictions", "","Y"], alpha = alpha_points, color = colors[i])
            gl.scatter(X_data_val_class[:,0].flatten(),  X_data_val_class[:,1].flatten(), ax = ax1, lw = 3,color = colors[i], #legend = ["val points"], 
                       alpha = alpha_points, marker = ">")

    out = ax1.contourf(xx, yy, most_likely_ygrid.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.5)
    
#    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    
#    gl.plot (xgrid_real_func, ygrid_real_func, ax  = ax1, alpha = 0.90, color = color_truth, legend = ["Truth"])
    for i in range(classes.size):
        X_data_tr_class = X_data_tr[np.where(Y_data_tr == classes[i])[0],:]
        X_data_val_class = X_data_val[np.where(Y_data_val == classes[i])[0],:]
#        print (X_data_tr_class.shape)
#        print (classes)
#        print (X_data_tr)
        if ((X_data_tr_class.size > 0) and (X_data_val_class.size > 0)):
            gl.scatter(X_data_tr_class[:,0].flatten().tolist(), X_data_tr_class[:,1].flatten().tolist(), ax = ax2, lw = 3,  #legend = ["tr points"], 
                       labels = ["", "X","Y"], alpha = alpha_points, color = colors[i])
            gl.scatter(X_data_val_class[:,0].flatten(),  X_data_val_class[:,1].flatten(), ax = ax2, lw = 3,color = colors[i], #legend = ["val points"], 
                       alpha = alpha_points, marker = ">")

    for ygrid in all_y_grid:
       out = ax2.contourf(xx, yy, ygrid.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.5)
    

    ############## ax2: Data + Realizations of the function !! ######################
    
    gl.set_zoom(xlimPad = [0.3,0.3], ylimPad = [0.3,0.3], ax = ax2, X = X_data_tr[:,0], Y = X_data_tr[:,1])


"""
####################### RNN EXAMPLE SPECIFIC ###########################3
"""
def compute_RNN_1D_data( model,X_data_tr,X_data_val, Nsamples = 100):
    """
    This function computes the outputs of the Regression model for the 1D example
    In this case we will just plot one sequence of validation
    """
    model.set_posterior_mean(False)
    device = model.cf_a.device
    dtype = model.cf_a.dtype
    ################ Obtain the data: ##################
    x_grid = X_data_val[[0], :]
    
    future = x_grid.shape[1]
    
    model.set_future(future)
    all_y_grid = []  ## Used to compute the variance of the prediction !!
    for i in range(Nsamples):
#        print (y_grid[0])
        y_grid =  model.predict(torch.tensor(x_grid,device=device, dtype=dtype)).detach().cpu().numpy()
        all_y_grid.append(y_grid)
        
    model.set_posterior_mean(True)
    most_likely_ygrid =  model.predict(torch.tensor(x_grid,device=device, dtype=dtype)).detach().cpu().numpy()
    
    return x_grid, all_y_grid,most_likely_ygrid

def plot_data_RNN_1d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                  x_grid,all_y_grid, most_likely_ygrid,
                                  alpha_points, color_points_train, color_points_val, color_most_likey,color_mean, color_truth,
                                  ax1,ax2):
    """
    This function plots the outputs of the Regression model for the 1D example
    """
    
    ## Compute mean and std of regression
    std_samples_grid = np.std(all_y_grid, axis = 1)
    mean_samples_grid = np.mean(all_y_grid, axis = 1)
    
    ############## ax1: Data + Mostlikely + Real + Mean !! ########################
#    gl.scatter(X_data_tr, Y_data_tr, ax = ax1, lw = 3,  #legend = ["tr points"], 
#               labels = ["Data and predictions", "","Y"], alpha = alpha_points, color = color_points_train)
    gl.plot([], x_grid.T, ax = ax1, lw = 2, ls = "--", #legend = ["val points"], 
               alpha = 0.8, color = color_points_train)
    
#    gl.plot (xgrid_real_func, ygrid_real_func, ax  = ax1, alpha = 0.90, color = color_truth, legend = ["Truth"])
    gl.plot ([], most_likely_ygrid.T, ax  = ax1, alpha = 0.90, color = color_most_likey, legend = ["Most likely"], ls = "--", lw = 2)

#    gl.plot ([], mean_samples_grid.T, ax  = ax1, alpha = 0.90, color = color_mean, legend = ["Posterior mean"],
#             AxesStyle = "Normal - No xaxis")
    
    ############## ax2: Data + Realizations of the function !! ######################
    gl.plot([], x_grid.T, ax = ax2, lw = 2, ls = "--", #legend = ["val points"], 
               alpha = 0.8, color = color_points_val)
    
    for y_grid_sample in all_y_grid:
        gl.plot([], y_grid_sample.T, ax = ax2, lw = 2, ls = "--", #legend = ["val points"], 
                   alpha = 0.3, color = "k")
#    gl.scatter(X_data_val, Y_data_val, ax = ax2, lw = 3, # legend = ["val points"], 
#               alpha = alpha_points, color = color_points_val)
    
    
#    gl.set_zoom(xlimPad = [0.3,0.3], ylimPad = [0.3,0.3], ax = ax2, X = , Y = Y_data_tr)

"""
Classification RNN name specific
"""
import matplotlib.ticker as ticker

    
def plot_confusion_matrix(confusion,all_categories, ax ):
    all_categories = list(all_categories)
    cax = ax.matshow(confusion)
    plt.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

"""
############################################################
INDIVIDUAL FUNCTIONS THAT PLOT INFORMATION IN PROVIDED AXES
############################################################
"""

def plots_weights_layer(mu_W, sigma_W,sigma_b, mu_b, ax1, ax2, legend_layer, plot_pdf = 1):
        """
        Plot the given weights of the layer
        """
        # For each of the weights we plot them !!
        color = gl.get_color(None) 
        if (plot_pdf):
            for i in range(sigma_W.size):
                x_grid, y_val = bMA.gaussian1D_points(mean = mu_W[i], std = sigma_W[i],  std_K = 3)
                
                gl.plot(x_grid, y_val, ax = ax1, fill = 1, alpha = 0.15, color = color,
                        labels = ["Bayesian weights","","p(w)"],alpha_line = 0.15   # ,legend = ["W:%i"%(i+1)]
                        )  ###legend = ["M: %.2e, std: %.2e"%(mu_W2[i], sigma_W2[i])])
            
        gl.scatter( mu_W, sigma_W, ax = ax2, labels = ["",r"$\mu_w$",r"$\sigma_w$"], 
                   color = color, legend = legend_layer, alpha = 0.3)
        if (plot_pdf):
            for i in range(sigma_b.size):
                x_grid, y_val = bMA.gaussian1D_points(mean = mu_b[i], std = sigma_b[i],  std_K = 3)
    #            color = gl.get_color(None)
                gl.plot(x_grid, y_val,  ax = ax1, color = color, fill = 1, alpha = 0.3, alpha_line = 0.15, AxesStyle = "Normal - No xaxis", ls = "--"
        #                  ,legend = ["b:%i"%(i+1)]
                          )  ###legend = ["M: %.2e, std: %.2e"%(mu_W2[i], sigma_W2[i])])
        gl.scatter(mu_b, sigma_b, ax = ax2,  color = color, marker = "s", alpha = 0.3)
            
            
def create_plot_variational_weights(model, ax1, ax2, plot_pdf = True):
    """
    This function plots the variational weights in the 2 axes given
    """
    l = 0
    for VBmodel in model.VBmodels:
        l+=1
        if (VBmodel.type_layer == "linear"):
            sigma_W = Vil.softplus(VBmodel.rho_weight).detach().cpu().numpy().flatten()
            mu_W = VBmodel.mu_weight.detach().cpu().numpy().flatten()
            
            sigma_b = Vil.softplus(VBmodel.rho_bias).detach().cpu().numpy().flatten()
            mu_b = VBmodel.mu_bias.detach().cpu().numpy().flatten()
        
            legend_final = ["Layer %i"%(l)]
            plots_weights_layer(mu_W, sigma_W,sigma_b, mu_b, ax1, ax2, legend_final, plot_pdf)
            
        else:
            sigma_W = Vil.softplus(VBmodel.rho_weight_ih).detach().cpu().numpy().flatten()
            mu_W = VBmodel.mu_weight_ih.detach().cpu().numpy().flatten()
            
            sigma_b = Vil.softplus(VBmodel.rho_bias_ih).detach().cpu().numpy().flatten()
            mu_b = VBmodel.mu_bias_ih.detach().cpu().numpy().flatten()
            
            legend_final = ["LSTM layer ih: %i"%(l)]
            plots_weights_layer(mu_W, sigma_W,sigma_b, mu_b, ax1, ax2, legend_final,plot_pdf)
            
            # Now the hidden weights
            sigma_W = Vil.softplus(VBmodel.rho_weight_hh).detach().cpu().numpy().flatten()
            mu_W = VBmodel.mu_weight_hh.detach().cpu().numpy().flatten()
            
            sigma_b = Vil.softplus(VBmodel.rho_bias_hh).detach().cpu().numpy().flatten()
            mu_b = VBmodel.mu_bias_hh.detach().cpu().numpy().flatten()
            
            legend_final = ["LSTM layer hh: %i"%(l)]
            plots_weights_layer(mu_W, sigma_W,sigma_b, mu_b, ax1, ax2, legend_final,plot_pdf)
            
        prior = VBmodel.prior
        gl.colorIndex -= 1; # So that the color is the same as the weights
        gl.scatter(0, prior.sigma1, lw = 3, ax = ax2, legend = ["Prior layer %i"%l], marker = "x")
    ### Plot the prior weights !! 
 
#    gl.scatter(0, prior.sigma2, lw = 3,ax = ax2, legend = ["Prior 2 (%.2f)"%(1-prior.pi_mix)], color = "b",marker = "x" ) 
    
    
"""
############################################################
FUNCTIONS THAT PLOT SOMETHING SMALL AND SAVE TO DISK
############################################################
"""
def plot_learnt_function(X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                          x_grid, y_grid, cf_a,
                          folder_images):
    gl.init_figure()
    ax1 = gl.scatter(X_data_tr, Y_data_tr, lw = 3,legend = ["tr points"], labels = ["Data", "X","Y"], alpha = 0.2)
    ax2 = gl.scatter(X_data_val, Y_data_val, lw = 3,legend = ["val points"], alpha = 0.2)
    
    gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 20, xticks = 12, yticks = 12)
    
    gl.plot (x_grid, y_grid, legend = ["training line"])
    gl.savefig(folder_images +'Training_Example_Data.png', 
               dpi = 100, sizeInches = [14, 4])
    
def plot_evolution_RMSE(tr_loss, val_loss, cf_a, folder_images):
    gl.init_figure()
    ax1 = gl.plot([], tr_loss, lw = 3, labels = ["RMSE loss and parameters. Learning rate: %.3f"%cf_a.lr, "","RMSE"], legend = ["train"])
    gl.plot([], val_loss, lw = 3, legend = ["validation"])
    
    
    gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 20, xticks = 12, yticks = 12)
    gl.savefig(folder_images +'Training_Example_Parameters.png', 
               dpi = 100, sizeInches = [14, 7])
    
def plot_weights_network(model, folder_images):

#
    weights = model.linear1.weight.detach().numpy()
    biases = model.linear1.bias.detach().numpy().reshape(-1,1)
    neurons = np.concatenate((weights, biases), axis = 1)
    
    weights2 = model.W2.detach().numpy()
    biases2 = model.b2.detach().numpy().reshape(-1,1)
    neurons2 = np.concatenate((weights2, biases2), axis =0).T
    
    gl.init_figure();
    ax1 = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=2)
    ax2 = gl.subplot2grid((1,4), (0,3), rowspan=1, colspan=4)

    cmap = cm.get_cmap('coolwarm', 30)
    cax = ax1.imshow(neurons, interpolation="nearest", cmap=cmap)
    cax2 = ax2.imshow(neurons2, interpolation="nearest", cmap=cmap)
    
#    plt.xticks(range(data_df_train.shape[1]), data_df_train.columns, rotation='vertical')
#    plt.yticks(range(data_df_train.shape[1]), data_df_train.columns, rotation='horizontal')
    plt.colorbar(cax)
#    plt.colorbar(cax2)
#        ax1.set_xticks(data_df_train.columns) # , rotation='vertical'
#    ax1.grid(True)
    plt.title('Weights ')
#    labels=[str(x) for x in range(Nshow )]
#    ax1.set_xticklabels(labels,fontsize=20)
#    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.show()
    gl.savefig(folder_images +'Weights.png', 
           dpi = 100, sizeInches = [2*8, 2*2])



"""
######################### OTHER AND OLD ##############################
"""


def create_image_training_epoch(X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                tr_loss, val_loss, x_grid, y_grid, cf_a,
                                video_fotograms_folder, epoch_i):
    """
    Creates the image of the training and validation accuracy
    """
    gl.init_figure();
    ax1 = gl.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((2,1), (1,0), rowspan=1, colspan=1)
    
    plt.title("Training")
    ## First plot with the data and predictions !!!
    ax1 = gl.scatter(X_data_tr, Y_data_tr, ax = ax1, lw = 3,legend = ["tr points"], labels = ["Analysis of training", "X","Y"])
    gl.scatter(X_data_val, Y_data_val, lw = 3,legend = ["val points"])
    
    gl.plot (x_grid, y_grid, legend = ["Prediction function"])

    gl.set_zoom(xlimPad = [0.2, 0.2], ylimPad = [0.2,0.2], X = X_data_tr, Y = Y_data_tr)
    ## Second plot with the evolution of parameters !!!
    ax2 = gl.plot([], tr_loss, ax = ax2, lw = 3, labels = ["RMSE. lr: %.3f"%cf_a.lr, "epoch","RMSE"], legend = ["train"])
    gl.plot([], val_loss, lw = 3, legend = ["validation"], loc = 3)
    
    
    gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 20, xticks = 12, yticks = 12)
    
    # Set final properties and save figure
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.30)
    
    gl.savefig(video_fotograms_folder +'%i.png'%epoch_i, 
               dpi = 100, sizeInches = [14, 10], close = True, bbox_inches = None)
        
    








