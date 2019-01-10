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
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.utils.data as data_utils

from BasicMLP import BasicMLP

"""
######################## OPTIONS ##############################
"""
folder_images = "../pics/Pytorch/BasicExamples/"
video_fotograms_folder  = "../pics/Pytorch/BasicExamples/video_1/"
video_fotograms_folder2  = "../pics/Pytorch/BasicExamples/video_2/"
folder_model = "../models/pyTorch/Basic0/"

#################### DATA AND DEVICES OPTIONS ############
linear_data = 0  # Linear or sinusoid data
dtype = torch.float
device = torch.device("cpu")
load_previous_state = 0
train_model = 1

################ PLOTTING OPTIONS ################
see_variables = 0
plot_predictions = 1
plot_evolution_loss =1
create_video_training = 1
plot_weights = 0


"""
###################### Setting up func ###########################
"""
## Set up seeds for more reproducible results
np.random.seed(0)
torch.manual_seed(0)

## Windows and folder management
plt.close("all") # Close all previous Windows
ul.create_folder_if_needed(folder_images)
ul.create_folder_if_needed(video_fotograms_folder)
ul.create_folder_if_needed(video_fotograms_folder2)
ul.create_folder_if_needed(folder_model)
if(create_video_training):
    ul.remove_files(video_fotograms_folder)
    ul.remove_files(video_fotograms_folder2)
"""
######################## LOAD AND PROCESS THE DATA ########################
"""
## Original Numpy data. The last data point is an outlier !!
if (linear_data):
    [X_data_tr, Y_data_tr, X_data_val,Y_data_val] = dl.get_linear_dataset(outlier = False)

else:
    Ntrain = 100; Nval = 50; sigma_noise = 0.1
    [X_data_tr, Y_data_tr, X_data_val,Y_data_val] = \
        dl.get_sinuoid_dataset(Ntrain = Ntrain, Nval = Nval, sigma_noise = sigma_noise)

## Normalize data:
k = 1.5
X_data_tr, X_data_val = dl.normalize_data(X_data_tr, X_data_val, k = k)
Y_data_tr, Y_data_val = dl.normalize_data(Y_data_tr, Y_data_val, k = k)

##############################################################
## Turn data into pyTorch Tensors !!
Xtrain = torch.tensor(X_data_tr,device=device, dtype=dtype)
Ytrain = torch.tensor(Y_data_tr,device=device, dtype=dtype)

Xval = torch.tensor(X_data_val,device=device, dtype=dtype)
Yval = torch.tensor(Y_data_val,device=device, dtype=dtype)

"""
######################## Config files ########################
"""
N, D_in = Xtrain.shape
N, D_out = Ytrain.shape
criterion = nn.MSELoss()

## LOAD THE BASE DATA CONFIG
cf_a = cf.cf_a

## Set data and device parameters
cf_a.D_in = D_in 
cf_a.D_out = D_out 

cf_a.dtype = dtype  # Variable types
cf_a.device = device

# Set other training parameters
cf_a.loss_func = criterion

"""
######################## CREATE DATA GENERATORS !! ########################
"""

#training_set = Dataset(Xtrain, Ytrain)
train = data_utils.TensorDataset(Xtrain, Ytrain)
train_loader = data_utils.DataLoader(train, batch_size=cf_a.batch_size, shuffle=True)

"""
######################## Instantiate Architecture ########################
"""
myBasicMLP = BasicMLP(cf_a)
# Set the model in training mode for the forward pass. self.training = True
# This is for Dropout and VI
myBasicMLP.train()  

if (load_previous_state):
    files_path = ul.get_allPaths(folder_model)
    if (len(files_path) >0):
        # Load the latest params !
        files_path = sorted(files_path, key = ul.cmp_to_key(ul.filenames_comp))
        myBasicMLP.load(files_path[-1])
    
"""
######################## Visualize variables ########################
Code that visualizes the architecture, data propagation and gradients.
"""

if (see_variables):
    print(myBasicMLP)
    myBasicMLP.print_parameters()
    myBasicMLP.print_parameters_names()
    myBasicMLP.print_named_parameters()
    myBasicMLP.print_gradients(Xtrain, Ytrain)
    
"""
######################## Optimizer !! ########################
Here we play with different optimizers for training
"""
optimizer_hidden = optim.SGD([myBasicMLP.linear1.weight,myBasicMLP.linear1.bias], lr = cf_a.lr)
optimizer_output = optim.SGD([myBasicMLP.W2, myBasicMLP.b2], lr= cf_a.lr)
optimizer_all = optim.SGD(myBasicMLP.parameters(), lr=cf_a.lr)

cf_a.op_h = optimizer_hidden
cf_a.op_o =  optimizer_output
cf_a.op_a = optimizer_all
"""
########################################################################
######################## TRAIN !! ########################
########################################################################
"""

training_mode = 3

if (train_model):
    tr_loss = []
    val_loss = []
    
    for i in range(cf_a.Nepochs):
        loss_tr_epoch = []
        for local_batch, local_labels in train_loader:
                if (training_mode == 1):
                    # All parameters are trained using SGD with lr and same loss function 
                    # without any optimizer
                    loss_i = myBasicMLP.train_batch(local_batch, local_labels).item()
                    
                elif (training_mode == 2):
                    # The output and input layer are trained with 2 different optimizers
                    # with the same loss function
                    loss_i = myBasicMLP.train_batch_optimi(local_batch, local_labels).item()
                elif (training_mode == 3):
                    # The output and input layer are trained with 2 different optimizers
                    # with different loss function
                    loss_i = myBasicMLP.train_batch_optimi_2(local_batch, local_labels).item()
        ## Get the loss for tr and val for each epoch.
        myBasicMLP.eval()
        tr_loss.append(myBasicMLP.get_loss(Xtrain, Ytrain).item())
        val_loss.append(myBasicMLP.get_loss(Xval, Yval).item())
        
        ##### CREATE VIDEO OF TRAINING ###############
        if (create_video_training):
            if (i%10 == 0):
                x_grid = np.linspace(np.min([X_data_tr]) -1, np.max([X_data_val]) +1, 100)
                y_grid =  myBasicMLP.predict(torch.tensor(x_grid.reshape(-1,1),device=cf_a.device, dtype=cf_a.dtype)).detach().numpy()
            
                pf.create_image_training_epoch(X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                        tr_loss, val_loss, x_grid, y_grid, cf_a,
                                        video_fotograms_folder, i)
                pf.create_image_weights_epoch(myBasicMLP, video_fotograms_folder2, i)
                
        myBasicMLP.train()
        
    ## Convert MSE to RMSE
    tr_loss = np.sqrt(tr_loss)
    val_loss = np.sqrt(val_loss)
        
if(create_video_training):  #
    pf.create_video_from_images(video_fotograms_folder,output_file = "./training_loss.avi", fps = 2)
    pf.create_video_from_images(video_fotograms_folder2,output_file = "./training_weights.avi", fps = 2)

"""
######################  SAVE MODEL ####################
"""
myBasicMLP.save(folder_model + "model_parameters_epoch:%i.pk"%i)

"""
#####################  STATIC PLOTS ###################### 
"""
# Set it in no training
myBasicMLP.eval()

if (plot_predictions):
    ####### PLOT THE LEARNT FUNCTION ############
    x_grid = np.linspace(np.min([X_data_tr]) -1, np.max([X_data_val]) +10, 10000)
    y_grid =  myBasicMLP.predict(torch.tensor(x_grid.reshape(-1,1),device=device, dtype=dtype)).detach().numpy()

    pf.plot_learnt_function(X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                     x_grid, y_grid, cf_a,
                                    folder_images)
    
if(plot_evolution_loss):
    ####### PLOT THE EVOLUTION OF RMSE ############
    pf.plot_evolution_RMSE(tr_loss, val_loss, cf_a,folder_images)

if (plot_weights):
    ####### PLOT ANALYSIS OF THE WEIGHTS ###########
    pf.plot_weights_network(myBasicMLP, folder_images)
