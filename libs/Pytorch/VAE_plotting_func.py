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

def create_Bayesian_analysis_charts(model,
                                    X_data_tr, X_data_val,
                                    tr_data_loss, val_data_loss, 
                                    KL_loss_tr, KL_loss_val, 
                                    final_loss_tr,final_loss_val,
                                    folder_images,
                                    epoch_i = None):

    # Configurations of the plots
   
    alpha_points = 0.2 
    color_points_train = "dark navy blue"
    color_points_val = "amber"
    color_truth = "k"
    color_mean = "b"
    color_most_likey = "y"



        
    ################################ Divide in plots ##############################
    gl.init_figure();
    ax1 = gl.subplot2grid((6,3), (0,0), rowspan=3, colspan=1)
    ax2 = gl.subplot2grid((6,3), (3,0), rowspan=3, colspan=1, sharex = ax1)
    
    ax3 = gl.subplot2grid((6,3), (0,1), rowspan=2, colspan=1)
    ax4 = gl.subplot2grid((6,3), (2,1), rowspan=2, colspan=1, sharex = ax3)
    ax5 = gl.subplot2grid((6,3), (4,1), rowspan=2, colspan=1, sharex = ax3)
    
    ax6 = gl.subplot2grid((6,3), (0,2), rowspan=3, colspan=1)
    ax7 = gl.subplot2grid((6,3), (3,2), rowspan=3, colspan=1, sharex = ax6)
    
   
    """
    ############################# Data computation #######################
    """

    Xtrain_sample_cpu, Xtrain_reconstruction,Xtrain_reconstruction_samples = \
        compute_reconstruction_data( model,X_data_tr, Nsamples = 100, sample_index = 2)

    plot_reconstruction_data (Xtrain_sample_cpu, Xtrain_reconstruction,Xtrain_reconstruction_samples,
                              ax1, ax2)
    """
    ############## ax3 ax4 ax5: Loss Evolution !! ######################
    """
    plot_losses_evolution_epoch(tr_data_loss, val_data_loss, 
                                        KL_loss_tr, KL_loss_val, 
                                        final_loss_tr,final_loss_val,
                                        ax3,ax4,ax5)
    
    """
    ############## ax6 ax7: Projecitons Weights !! ######################
    """
    plot_projections_VAE(model,X_data_tr,ax6)
    ## Plot in chart 7 the acceptable mu = 2sigma  -> sigma = |mu|/2sigma 

#    gl.set_zoom (ax = ax6, ylim = [-0.1,10])
#    gl.set_zoom (ax = ax7, xlim = [-2.5, 2.5], ylim = [-0.05, np.exp(model.cf_a.input_layer_prior["log_sigma2"])*(1 + 0.15)])
    
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
    

def plot_losses_evolution_epoch(tr_data_loss, val_data_loss, 
                                    KL_loss_tr, KL_loss_val, 
                                    final_loss_tr,final_loss_val,
                                    ax1,ax2,ax3):
    color_train_loss = "cobalt blue"
    color_val_loss = "blood"
    ############## ax3 ax4 ax5: Loss Evolution !! ######################
    ## ax3: Evolutoin of the data loss
    gl.plot([], tr_data_loss, ax = ax1, lw = 3, labels = ["Losses", "","Data loss"], legend = ["train"],
            color = color_train_loss)
    gl.plot([], val_data_loss,ax = ax1, lw = 3, legend = ["validation"],
            color = color_val_loss,  AxesStyle = "Normal - No xaxis")
    
    ## ax4: The evolution of the KL loss
    gl.plot([], KL_loss_tr, ax = ax2, lw = 3, labels = ["", "","KL loss"], legend = ["train"],
            AxesStyle = "Normal - No xaxis", color = color_train_loss)
    gl.plot([], KL_loss_val, ax = ax2, lw = 3, labels = ["", "","KL loss"], legend = ["validation"],
            AxesStyle = "Normal - No xaxis", color = color_val_loss)

    ## ax5: Evolutoin of the total loss
    gl.plot([], final_loss_tr, ax = ax3, lw = 3, labels = ["", "epoch","Total Loss (Bayes)"], legend = ["train"],
            color = color_train_loss)
    gl.plot([], final_loss_val,ax = ax3, lw = 3, legend = ["validation"], color = color_val_loss)
    
    
    
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
def compute_reconstruction_data( model,X_data_tr, 
                               Nsamples = 100, sample_index = 2):
    """
    This function computes the outputs of the reconstruction of a given sample
    """
    device = model.cf_a.device
    dtype = model.cf_a.dtype
    
    
    ################ Obtain the data: ##################

    model.eval()
    Xtrain_sample = torch.tensor( X_data_tr[sample_index,:],device=device, dtype=dtype) 
    Xtrain_reconstruction = model.forward(Xtrain_sample)[0].detach().cpu().numpy()
    
    Xtrain_sample_cpu = Xtrain_sample.detach().cpu().numpy()

    ### SEVERAL OF THE transformations ######
    Xtrain_reconstruction_samples = []
    model.train()
    for i in range(300):
        Xtrain_reconstruction_sample = model.forward(Xtrain_sample)[0].detach().cpu().numpy().reshape(1,-1)
        Xtrain_reconstruction_samples.append(Xtrain_reconstruction_sample)
    Xtrain_reconstruction_samples = np.concatenate(Xtrain_reconstruction_samples, axis = 0).T
    
    print (Xtrain_sample_cpu.shape, Xtrain_reconstruction.shape, Xtrain_reconstruction_samples.shape)
    return Xtrain_sample_cpu, Xtrain_reconstruction,Xtrain_reconstruction_samples

def plot_reconstruction_data (Xtrain_sample_cpu, Xtrain_reconstruction,Xtrain_reconstruction_samples,
                              ax1, ax2):

    title = "Original sample and reconstructions"
    
    x = []
    alpha = 0.9
    cumulated_samples = 0
    
    gl.plot(x,Xtrain_sample_cpu , ax = ax1,legend = ["Original"] , color = "k",alpha = alpha,
                    labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis")
    
    gl.plot(x,Xtrain_reconstruction , ax = ax1,legend = ["Reconstruction"]  , color = "b",alpha = alpha,
                    labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis")
    
    ### SEVERAL OF THE transformations ######
    gl.plot(x,Xtrain_sample_cpu , ax = ax2,legend = ["Reconstruction"] , color = "k",alpha = alpha,
                    labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis")
    
    alpha  = 0.2

    gl.plot(x,Xtrain_reconstruction_samples , ax = ax2,legend = []  , color = "b",alpha = alpha,
                    labels = [title,"",r"Rate"], AxesStyle = "Normal - No xaxis")
    
    

"""
####################### CLASSIFICATION EXAMPLE SPECIFIC ###########################3
"""
from sklearn.decomposition import PCA

def plot_projections_VAE(model,X_data_tr,ax1):
    """
    This function should plot the PCA mus to 2 Dimensions of the space
    """
    device = model.cf_a.device
    dtype = model.cf_a.dtype
    
    Xtrain = torch.tensor(X_data_tr,device=device, dtype=dtype)

    model.eval()
    
    reconstructions, mus, rhos = model.forward(Xtrain)
    
    ## We have the mus and std of every sample !!
    mus = mus.detach().cpu().numpy()  ## [Nsamples, Dim]
    std = Vil.softplus(rhos).detach().cpu().numpy()

    pca = PCA(n_components=2)
    pca.fit(mus)

    ## PCA projection to 2 Dimensions 
    ## TODO, for now we just get the 2 variables
    mus_to_2_dim = pca.transform(mus) #mus[:,[0,1]]  
    stds_to_2_dim = std[:,[0,1]]  
    
    alpha = 0.2
    
    target = X_data_tr[:,-1] - X_data_tr[:,0]

    ###### COLOR #########
    for i in range (X_data_tr.shape[0]):
        if target[i] > 0:
            gl.scatter(mus_to_2_dim[i,0], mus_to_2_dim[i,1], legend = [], labels = ["Projeciton of points","var1","var2"],
               alpha = alpha, ax = ax1, color = "r")
        else:
            gl.scatter(mus_to_2_dim[i,0], mus_to_2_dim[i,1], legend = [], labels = ["Projeciton of points","var1","var2"],
               alpha = alpha, ax = ax1, color = "b")  

#        total_variance = stds_to_2_dim[i,0]*stds_to_2_dim[i,1]
#        circle1 = plt.Circle((mus_to_2_dim[i,0], mus_to_2_dim[i,1]), total_variance, color='k')
#        ax1.add_artist(circle1)

def plot_mus_std(model,X_data_tr,ax1):
    """
    This function should plot the PCA mus to 2 Dimensions of the space
    """
    device = model.cf_a.device
    dtype = model.cf_a.dtype
    
    Xtrain = torch.tensor(X_data_tr,device=device, dtype=dtype)

    model.eval()
    
    reconstructions, mus, rhos = model.forward(Xtrain)
    
    ## We have the mus and std of every sample !!
    mus = mus.detach().cpu().numpy()  ## [Nsamples, Dim]
    std = Vil.softplus(rhos).detach().cpu().numpy()
    
    ## PCA projection to 2 Dimensions 
    ## TODO, for now we just get the 2 variables
    
    
    
    mus_to_2_dim = mus[:,[0,1]]  
    
    alpha = 0.2
    
    target = X_data_tr[:,-1] - X_data_tr[:,0]
    
    ###### COLOR #########
    for i in range (X_data_tr.shape[0]):
        if target[i] > 0:
            gl.scatter(mus_to_2_dim[i,0], mus_to_2_dim[i,1], legend = [], labels = ["Projeciton of points","var1","var2"],
               alpha = alpha, ax = ax1, color = "r")
        else:
            gl.scatter(mus_to_2_dim[i,0], mus_to_2_dim[i,1], legend = [], labels = ["Projeciton of points","var1","var2"],
               alpha = alpha, ax = ax1, color = "b")  
        
#    gl.scatter(mus_to_2_dim[:,0], mus_to_2_dim[:,1], legend = ["Points"], labels = ["Projeciton of points","var1","var2"],
#               alpha = alpha, ax = ax1)
#    
    
    
    
    
    
    

