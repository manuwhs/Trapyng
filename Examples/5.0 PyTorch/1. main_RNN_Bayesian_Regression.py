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
import pyTorch_utils as pytut
from GeneralVBModelRNN import GeneralVBModelRNN
from RNN_regression_1D_fullVB import RNN_regression_1D_fullVB
import Variational_inferences_lib as Vil

"""
######################## OPTIONS ##############################
"""
folder_images = "../pics/Pytorch/BasicExamples_Bayesian_RNN_Regression_gpu/"
video_fotograms_folder  = folder_images +"video_Bayesian1/"
video_fotograms_folder2  = folder_images +"video_Bayesian2/"
video_fotograms_folder3 = folder_images +"video_Bayesian3/"
video_fotograms_folder4 = folder_images +"video_Bayesian4/"
folder_model = "../models/pyTorch/Basic0_Bayesian_RNN/"

#################### DATA AND DEVICES OPTIONS ############
linear_data = 0  # Linear or sinusoid data
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)
load_previous_state = 0
train_model = 1

################ PLOTTING OPTIONS ################
see_variables = 0
plot_predictions = 1
plot_evolution_loss =1
create_video_training = 1
plot_weights = 0
Step_video = 1

"""
###################### Setting up func ###########################
"""
## Set up seeds for more reproducible results
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
## Windows and folder management
plt.close("all") # Close all previous Windows
ul.create_folder_if_needed(folder_images)
ul.create_folder_if_needed(video_fotograms_folder)
ul.create_folder_if_needed(video_fotograms_folder2)
ul.create_folder_if_needed(video_fotograms_folder3)
ul.create_folder_if_needed(video_fotograms_folder4)
ul.create_folder_if_needed(folder_model)

if(create_video_training):
    ul.remove_files(video_fotograms_folder)
    ul.remove_files(video_fotograms_folder2)
    ul.remove_files(video_fotograms_folder3)
    ul.remove_files(video_fotograms_folder4)
"""
######################## LOAD AND PROCESS THE DATA ########################
"""
# load data and make training set
data = dl.generate_sine_data(T = 20, L = 400,  N = 100)
data = dl.load_RNN_data_generated()
xgrid_real_func = None; ygrid_real_func = None; 

# We train with the first 97 chains and validate with the final 3
X_data_tr = data[3:, :-1];
Y_data_tr =data[3:, 1:]

X_data_val = data[:3, :-1]
Y_data_val =  data[:3, 1:]

##############################################################
## Turn data into pyTorch Tensors !!
Xtrain = torch.tensor(X_data_tr,device=device, dtype=dtype)
Ytrain = torch.tensor(Y_data_tr,device=device, dtype=dtype)

Xval = torch.tensor(X_data_val,device=device, dtype=dtype)
Yval = torch.tensor(Y_data_val,device=device, dtype=dtype)

"""
######################## Config files ########################
"""
Ntrain, T = Xtrain.shape
criterion = nn.MSELoss()

## LOAD THE BASE DATA CONFIG
cf_a = cf.cf_RNN_1

## Set data and device parameters
cf_a.D_in = 1 
cf_a.D_out = 1 

cf_a.dtype = dtype  # Variable types
cf_a.device = device

# Set other training parameters
cf_a.loss_func = criterion

#cf_a.batch_size = Ntrain

"""
######################## Bayesian Prior ! ########################
"""
log_sigma_p1 = np.log(0.1)
log_sigma_p2 = np.log(0.3)
pi = 0.5
prior = Vil.Prior(pi, log_sigma_p1,log_sigma_p2)

"""
######################## CREATE DATA GENERATORS !! ########################
"""

#training_set = Dataset(Xtrain, Ytrain)
train = data_utils.TensorDataset(Xtrain, Ytrain)
train_loader = data_utils.DataLoader(train, batch_size=cf_a.batch_size, shuffle=True)

"""
######################## Instantiate Architecture ########################
"""
#myGeneralVBModel = GeneralVBModelRNN(cf_a, prior)
myGeneralVBModel = RNN_regression_1D_fullVB(cf_a, prior)

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
######################## Optimizer !! ########################
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
    cf_a.optimizer_params = dict([["lr", 0.1]])
    cf_a.optimizer = optim.SGD(parameters, lr= cf_a.optimizer_params["lr"])
elif(optimizer_type == "Adam"):
    cf_a.optimizer_params = dict([["lr", 0.01]])
    cf_a.optimizer = optim.Adam(parameters)
elif(optimizer_type == "LBFGS"):
    cf_a.optimizer_params = dict([["lr", 0.01]])
    cf_a.optimizer = optim.LBFGS(parameters, lr=0.8)
    
    
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
            KL_loss.append(-myGeneralVBModel.get_KL_loss().item())
            final_loss_tr.append(myGeneralVBModel.get_loss(Xtrain, Ytrain).item())
            final_loss_val.append(myGeneralVBModel.get_loss(Xval, Yval).item())
            
            if (create_video_training):
#                x_grid = np.linspace(np.min([X_data_tr]) -1, np.max([X_data_val]) +1, 100)
#                y_grid =  myHalfBayesianMLP.predict(torch.tensor(x_grid.reshape(-1,1),device=cf_a.device, dtype=cf_a.dtype),
#                                                    posterior_mean = True).detach().numpy()
#            
#                pf.create_image_training_epoch(X_data_tr, Y_data_tr, X_data_val, Y_data_val,
#                                        tr_loss, val_loss, x_grid, y_grid, cf_a,
#                                        video_fotograms_folder, i)
                pf.create_image_weights_epoch(myGeneralVBModel, video_fotograms_folder2, i)
#                pf.create_image_Variational_weights_network(myHalfBayesianMLP, video_fotograms_folder3, i)
            
                pf.create_Bayesian_analysis_charts(myGeneralVBModel,
                                                    X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                                    tr_data_loss, val_data_loss, KL_loss,
                                                    xgrid_real_func, ygrid_real_func,
                                                    video_fotograms_folder4, i)

        ## Train one Epoch !!!
        myGeneralVBModel.train()  
        myGeneralVBModel.set_posterior_mean(False)
        myGeneralVBModel.set_future(0) # Do not predict more than the input
        for local_batch, local_labels in train_loader:
            # The output and input layer are trained with 2 different optimizers
            # with different loss function
            loss_i = myGeneralVBModel.train_batch(local_batch, local_labels).item()
            
        ## Get the loss for tr and val for each epoch.
        myGeneralVBModel.eval()
        myGeneralVBModel.set_posterior_mean(True)
        tr_data_loss.append(myGeneralVBModel.get_data_loss(Xtrain, Ytrain).item())
        val_data_loss.append(myGeneralVBModel.get_data_loss(Xval, Yval).item())
        KL_loss.append(-myGeneralVBModel.get_KL_loss().item())
        final_loss_tr.append(myGeneralVBModel.get_loss(Xtrain, Ytrain).item())
        final_loss_val.append(myGeneralVBModel.get_loss(Xval, Yval).item())
                
        ##### CREATE VIDEO OF TRAINING ###############
        if (create_video_training):
            if ((i+1)%Step_video == 0):
                myGeneralVBModel.eval()
                
#                x_grid = np.linspace(np.min([X_data_tr]) -1, np.max([X_data_val]) +1, 100)
#                y_grid =  myHalfBayesianMLP.predict(torch.tensor(x_grid.reshape(-1,1),device=cf_a.device, dtype=cf_a.dtype),
#                                                    posterior_mean = True).detach().numpy()
#            
#                pf.create_image_training_epoch(X_data_tr, Y_data_tr, X_data_val, Y_data_val,
#                                        tr_loss, val_loss, x_grid, y_grid, cf_a,
#                                        video_fotograms_folder, i+1)
                pf.create_image_weights_epoch(myGeneralVBModel, video_fotograms_folder2, i+1)
#                pf.create_image_Variational_weights_network(myHalfBayesianMLP, video_fotograms_folder3, i+1)
                pf.create_Bayesian_analysis_charts(myGeneralVBModel,
                                                    X_data_tr, Y_data_tr, X_data_val, Y_data_val,
                                                    tr_data_loss, val_data_loss, KL_loss,
                                                    xgrid_real_func, ygrid_real_func,
                                                    video_fotograms_folder4, i+1)
               
                

if(create_video_training):  #
#    pf.create_video_from_images(video_fotograms_folder,output_file = "./training_loss.avi", fps = 2)
    pf.create_video_from_images(video_fotograms_folder2,output_file =folder_images +  "/training_weights.avi", fps = 2)
#    pf.create_video_from_images(video_fotograms_folder3,output_file = "./training_Variational_weights.avi", fps = 2)
    
    pf.create_video_from_images(video_fotograms_folder4,output_file = folder_images + "/training_Variational_weights.avi", fps = 2)
    
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
                                        tr_data_loss, val_data_loss, KL_loss,
                                        xgrid_real_func, ygrid_real_func,
                                        folder_images)

if (plot_weights):
    ####### PLOT ANALYSIS OF THE WEIGHTS ###########
    pf.plot_weights_network(myGeneralVBModel, folder_images)
    pf.plot_Variational_weights_network(myGeneralVBModel, folder_images)
    
    

#begin to train
for i in range(15):
    print('STEP: ', i)

    # begin to predict, no need to track gradient here
    with torch.no_grad():
        future = 1000
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
