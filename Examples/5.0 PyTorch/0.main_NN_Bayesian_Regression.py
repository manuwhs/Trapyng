# -*- coding: utf-8 -*-
"""
"""

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import plotting_func as pf
# Specific utilities
import utilities_lib as ul

import data_loaders as dl
import config_files as cf 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import pyTorch_utils as pytut
from GeneralVBModel import GeneralVBModel
import Variational_inferences_lib as Vil

"""
######################## OPTIONS ##############################
"""
#################### DATA AND DEVICES OPTIONS ############
linear_data = 0  # Linear or sinusoid data
dtype = torch.float
#device = torch.device("cpu")
device = pytut.get_device_name(cuda_index = 0)

load_previous_state = 0
train_model = 1

################ PLOTTING OPTIONS ################
see_variables = 1
plot_predictions = 1
plot_evolution_loss =1

create_video_training = 1
plot_weights = 0
Step_video = cf.cf_a_Regression_1D.Nepochs

## We can do this for more than 1 hyperparameter
eta_values = [0, 0.01,0.05, 0.1,0.2,0.5,1,2,5]
eta_values = [0, 0.02,0.2]

saved_models = []
for eta_i in range(len(eta_values)):
    
    eta_KL = eta_values[eta_i]
    folder_images = "../pics/Pytorch/BasicExamples_Bayesian/MLP/"+ str(eta_KL) + "/"
    video_fotograms_folder_weights  = folder_images +"Weights/"
    video_fotograms_folder_training  = folder_images +"Training/"
    folder_model = "../models/pyTorch/BasicExamples_Bayesian/MLP/"
    
    """
    ###################### Setting up func ###########################
    """
    ## Set up seeds for more reproducible results
    random_seed = 0 #time_now.microsecond
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    ## Windows and folder management
    plt.close("all") # Close all previous Windows
    ul.create_folder_if_needed(folder_images)
    ul.create_folder_if_needed(video_fotograms_folder_weights)
    ul.create_folder_if_needed(video_fotograms_folder_training)
    ul.create_folder_if_needed(folder_model)
    
    if(create_video_training):
        ul.remove_files(video_fotograms_folder_weights)
        ul.remove_files(video_fotograms_folder_training)
    """
    ######################## LOAD AND PROCESS THE DATA ########################
    """
    ## Original Numpy data. The last data point is an outlier !!
    if (linear_data):
        [X_data_tr, Y_data_tr, X_data_val,Y_data_val] = dl.get_linear_dataset(outlier = False)
    
    else:
        Ncycles = 1.5
        Ntrain = 200; Nval = 50; sigma_noise = 0.15
        [X_data_tr, Y_data_tr, X_data_val,Y_data_val] = \
            dl.get_sinuoid_dataset(Ntrain = Ntrain, Nval = Nval, sigma_noise = sigma_noise, Ncycles = Ncycles)
        
        xgrid_real_func,ygrid_real_func, _,_ = \
            dl.get_sinuoid_dataset(Ntrain = Ntrain, Nval = Nval, sigma_noise = 0.0, Ncycles = Ncycles)
            
    ## Normalize data:
    k = 1
    X_data_val = dl.normalize_data(X_data_tr, X_data_val, k = k)
    Y_data_val = dl.normalize_data(Y_data_tr, Y_data_val, k = k)
    
    xgrid_real_func = dl.normalize_data(X_data_tr, xgrid_real_func, k = k)
    ygrid_real_func = dl.normalize_data(Y_data_tr, ygrid_real_func, k = k)
    
    X_data_tr = dl.normalize_data(X_data_tr, X_data_tr, k = k)
    Y_data_tr = dl.normalize_data(Y_data_tr, Y_data_tr, k = k)
    
    
    ##############################################################
    ## Turn data into pyTorch Tensors !!
    Xtrain = torch.tensor(X_data_tr,device=device, dtype=dtype)
    Ytrain = torch.tensor(Y_data_tr,device=device, dtype=dtype)
    
    Xval = torch.tensor(X_data_val,device=device, dtype=dtype)
    Yval = torch.tensor(Y_data_val,device=device, dtype=dtype)
    
    """
    ######################## Config files ########################
    """
    Ntr, D_in = Xtrain.shape
    Nval, D_out = Ytrain.shape
    criterion = nn.MSELoss()
    
    ## LOAD THE BASE DATA CONFIG
    cf_a = cf.cf_a_Regression_1D
    
    ## Set data and device parameters
    cf_a.D_in = D_in 
    cf_a.D_out = D_out 
    
    cf_a.dtype = dtype  # Variable types
    cf_a.device = device
    
    # Set other training parameters
    cf_a.loss_func = criterion
    
    """
    ######################## Bayesian Prior ! ########################
    """
    cf_a.eta_KL = eta_KL
    input_layer_prior  = cf_a.input_layer_prior
    output_layer_prior  = cf_a.output_layer_prior
    prior_example = Vil.Prior(**input_layer_prior)
    
    """
    ######################## CREATE DATA GENERATORS !! ########################
    """
    
    #training_set = Dataset(Xtrain, Ytrain)
    train = data_utils.TensorDataset(Xtrain, Ytrain)
    train_loader = data_utils.DataLoader(train, batch_size=cf_a.batch_size_train, shuffle=True)
    
    """
    ######################## Instantiate Architecture ########################
    """
    cf_a.Nsamples_train = Ntr
    
    myGeneralVBModel = GeneralVBModel(cf_a).to(device = device)
    # Set the model in training mode for the forward pass. self.training = True
    # This is for Dropout and VI
    myGeneralVBModel.train()  
    
    if (load_previous_state):
        files_path = ul.get_allPaths(folder_model)
        if (len(files_path) >0):
            # Load the latest params !
            files_path = sorted(files_path, key = ul.cmp_to_key(ul.filenames_comp))
            myGeneralVBModel.load(files_path[-1])
        
    """
    ######################## Visualize variables ########################
    Code that visualizes the architecture, data propagation and gradients.
    """
    if (see_variables):
        print(myGeneralVBModel)
        myGeneralVBModel.print_parameters()
        myGeneralVBModel.print_parameters_names()
        myGeneralVBModel.print_named_parameters()
        myGeneralVBModel.print_gradients(Xtrain, Ytrain)
        
    """
    ######################## Optimizer examaple!! ########################
    Here we play with different optimizers for training
    """
    #optimizer_hidden = optim.SGD([myGeneralVBModel.linear1.weight,myGeneralVBModel.linear1.bias], lr = cf_a.lr)
    #optimizer_output = optim.SGD([myHalfBayesianMLP.W2, myHalfBayesianMLP.b2], lr= cf_a.lr)
    #optimizer_all = optim.SGD(myGeneralVBModel.parameters(), lr=cf_a.lr)
    
    #cf_a.op_h = optimizer_hidden
    #cf_a.op_o =  optimizer_output
    #cf_a.op_a = optimizer_all
    
    optimizer_type = "SGD"
    parameters = myGeneralVBModel.parameters()
    if (optimizer_type == "SGD"):
        optimizer_params = dict([["lr", 0.1]])
        optimizer = optim.SGD(parameters, lr=cf_a.optimizer_params["lr"])
    elif(optimizer_type == "Adam"):
        optimizer_params = dict([["lr", 0.1]])
        optimizer = optim.Adam(parameters)
        
    """
    ########################################################################
    ######################## TRAIN !! ########################
    ########################################################################
    """
    
    myGeneralVBModel.eval()
    if (train_model):
        tr_data_loss = []
        val_data_loss = [] 
        
        KL_loss = []
        final_loss_tr = []
        final_loss_val = []
    
        for i in range(cf_a.Nepochs):
            loss_tr_epoch = []
    
            if (i == 0):
                ## First iteration we save the initialized values
                myGeneralVBModel.eval()
                myGeneralVBModel.set_posterior_mean(True)
                tr_data_loss.append(myGeneralVBModel.get_data_loss(Xtrain, Ytrain).item())
                val_data_loss.append(myGeneralVBModel.get_data_loss(Xval, Yval).item())
                KL_loss.append(myGeneralVBModel.get_KL_loss().item())
                final_loss_tr.append(myGeneralVBModel.get_loss(Xtrain, Ytrain).item())
                final_loss_val.append(myGeneralVBModel.get_loss(Xval, Yval).item())
                
                if (create_video_training):
    #                pf.create_image_weights_epoch(myGeneralVBModel, video_fotograms_folder_weights, i)
                    pf.create_Bayesian_analysis_charts(myGeneralVBModel,
                                                        X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                                        tr_data_loss, val_data_loss, KL_loss,final_loss_tr,final_loss_val,
                                                        xgrid_real_func, ygrid_real_func,
                                                        video_fotograms_folder_training, i)
    
            ## Train one Epoch !!!
            myGeneralVBModel.train()  
            myGeneralVBModel.set_posterior_mean(False)
            for local_batch, local_labels in train_loader:
                # The output and input layer are trained with 2 different optimizers
                # with different loss function
                loss_i = myGeneralVBModel.train_batch(local_batch, local_labels).item()
                
            ## Get the loss for tr and val for each epoch.
            myGeneralVBModel.eval()
            myGeneralVBModel.set_posterior_mean(True)
            tr_data_loss.append(myGeneralVBModel.get_data_loss(Xtrain, Ytrain).item())
            val_data_loss.append(myGeneralVBModel.get_data_loss(Xval, Yval).item())
            KL_loss.append(myGeneralVBModel.get_KL_loss().item())
            final_loss_tr.append(myGeneralVBModel.get_loss(Xtrain, Ytrain).item())
            final_loss_val.append(myGeneralVBModel.get_loss(Xval, Yval).item())
                    
            ##### CREATE VIDEO OF TRAINING ###############
            if (create_video_training):
                if ((i+1)%Step_video == 0):
                    myGeneralVBModel.eval()
    #                pf.create_image_weights_epoch(myGeneralVBModel, video_fotograms_folder_weights, i+1)
                    pf.create_Bayesian_analysis_charts(myGeneralVBModel,
                                                        X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                                        tr_data_loss, val_data_loss, KL_loss,final_loss_tr,final_loss_val,
                                                        xgrid_real_func, ygrid_real_func,
                                                        video_fotograms_folder_training, i+1)
                   
                    
    if(create_video_training):  #
        pf.create_video_from_images(video_fotograms_folder_weights,output_file =folder_images +  "/training_weights.avi", fps = 2)
        pf.create_video_from_images(video_fotograms_folder_training,output_file = folder_images + "/training_Variational_weights.avi", fps = 2)
        
    """
    ######################  SAVE MODEL ####################
    """
    myGeneralVBModel.save(folder_model + "model_parameters_epoch:%i.pk"%i)
    
    """
    #####################  STATIC PLOTS ###################### 
    """
    # Set it in no training
    myGeneralVBModel.eval()
    
    if (plot_predictions):
        ####### PLOT THE LEARNT FUNCTION ############
        pf.create_Bayesian_analysis_charts(myGeneralVBModel,
                                            X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                            tr_data_loss, val_data_loss, KL_loss,final_loss_tr,final_loss_val,
                                            xgrid_real_func, ygrid_real_func,
                                            folder_images)
    if (plot_weights):
        ####### PLOT ANALYSIS OF THE WEIGHTS ###########
        pf.plot_weights_network(myGeneralVBModel, folder_images)
        pf.plot_Variational_weights_network(myGeneralVBModel, folder_images)
    
    saved_models.append(myGeneralVBModel)
if (1):
        ### USED TO GENERATE THE CHART FOR THE INTRO:
    gl.init_figure()
    Nmodels = len(saved_models)
    all_axes = []
    for i in range(Nmodels):
        gl.colorIndex = 0
        model = saved_models[i]
        ax1 = gl.subplot2grid((2,Nmodels),(0,i), rowspan=1, colspan=1)
        ax2 = gl.subplot2grid((2,Nmodels),(1,i), rowspan=1, colspan=1)
        
        alpha_points = 0.2 
        color_points_train = "dark navy blue"
        color_points_val = "amber"
        color_train_loss = "cobalt blue"
        color_val_loss = "blood"
        color_truth = "k"
        color_mean = "b"
        color_most_likey = "y"
        
        x_grid, all_y_grid,most_likely_ygrid = pf.compute_regression_1D_data( model,X_data_tr,X_data_val, Nsamples = 100)
        pf.plot_data_regression_1d_2axes(X_data_tr, Y_data_tr, xgrid_real_func, ygrid_real_func, X_data_val, Y_data_val,
                                  x_grid,all_y_grid, most_likely_ygrid,
                                  alpha_points, color_points_train, color_points_val, color_most_likey,color_mean, color_truth,
                                  None,ax1)
        
        pf.create_plot_variational_weights(model, ax1, ax2, plot_pdf = False)
        all_axes.append(ax1);all_axes.append(ax2);
        
        mu_grid = np.linspace(-3,3,100)
        y_grid = np.abs(mu_grid)/2
        
        gl.fill_between(mu_grid, 10*np.ones(mu_grid.size), y_grid,
                        alpha = 0.2, color = "r", ax = ax2, legend = ["95% non-significant"])
        
        gl.set_zoom (ax = ax2, xlim = [-2.5, 2.5], ylim = [-0.05, model.linear1.prior.sigma1*(1 + 0.30)])
        
        eta_KL = eta_values[i]
        ax1.set_title("Model estimations for $\zeta = " + str(eta_KL)+ "$")
        ax2.set_title("Variational Weights for $\zeta = " + str(eta_KL)+ "$")
#    gl.set_zoom (ax = ax7, xlim = [-2.5, 2.5], ylim = [-0.1,2])
    
    # Set final properties and save figure
    gl.set_fontSizes(ax = all_axes, title = 14, xlabel = 16, ylabel = 16, 
                      legend = 10, xticks = 10, yticks = 10)


    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.25, hspace=0.25)
    

    gl.savefig("../pics/Pytorch/BasicExamples_Bayesian/MLP/Introudction_example.png", 
                   dpi = 100, sizeInches = [14, 9], close = True, bbox_inches = "tight")