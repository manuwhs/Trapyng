"""
Using our structure in the problem:
    
    https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    
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
import pyTorch_utils as pytut

from RNN_names_classifier import RNN_names_classifier
from RNN_names_classifier_fullVB import RNN_names_classifier_fullVB
import Variational_inferences_lib as Vil

"""
######################## OPTIONS ##############################
"""
folder_images = "../pics/Pytorch/BasicExamples_Bayesian_RNN_class_gpu_fullVB/"
video_fotograms_folder  = folder_images +"video_Bayesian1/"
video_fotograms_folder2  = folder_images +"video_Bayesian2/"
video_fotograms_folder3 = folder_images +"video_Bayesian3/"
video_fotograms_folder4 = folder_images +"video_Bayesian4/"
folder_model = "../models/pyTorch/Basic0_Bayesian_RNN/"

#################### DATA AND DEVICES OPTIONS ############
linear_data = 0  # Linear or sinusoid data
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)
save_model_per_epochs = 1
load_previous_state = 0
train_model = 1

################ PLOTTING OPTIONS ################
see_variables = 0
plot_predictions = 0
plot_evolution_loss =0
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

if(0):
    ul.remove_files(video_fotograms_folder)
    ul.remove_files(video_fotograms_folder2)
    ul.remove_files(video_fotograms_folder3)
    ul.remove_files(video_fotograms_folder4)
"""
######################## LOAD AND PROCESS THE DATA ########################
"""
# load data and make training set
xgrid_real_func = None; ygrid_real_func = None; 
[all_categories, X_data_tr, Y_data_tr, X_data_val,Y_data_val]  = dl.load_names_dataset(filepath = '../data/RNN_text/names/*.txt')

Y_data_tr2 = np.array(Y_data_tr)
##############################################################
## Turn data into pyTorch Tensors !!
Xtrain = [torch.tensor(X_data_tr[i],device=device, dtype=dtype) for i in range(len(X_data_tr))]
Ytrain = torch.tensor(Y_data_tr,device=device, dtype=torch.int64)

Xval = [torch.tensor(X_data_val[i],device=device, dtype=dtype) for i in range(len(X_data_val))]
Yval = torch.tensor(Y_data_val,device=device, dtype=torch.int64)

"""
######################## Config files ########################
"""
Ntrain = len(Xtrain)
T_i,nbatch_i,D_in = Xtrain[0].shape
criterion = nn.CrossEntropyLoss()

## LOAD THE BASE DATA CONFIG
cf_a = cf.cf_RNN_2

## Set data and device parameters
cf_a.D_in = D_in
cf_a.D_out = len(all_categories)

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

from torch.utils import data

class Dataset_RNN(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, inputs, labels ):
        'Initialization'
        self.labels = labels
        self.inputs = inputs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = torch.tensor(self.inputs[index],device=device, dtype=dtype)
        y = self.labels[index].reshape(1)

        return X, y
    
# Generators
training_set = Dataset_RNN(Xtrain, Ytrain)
train_loader = data.DataLoader(training_set, batch_size=cf_a.batch_size, shuffle=True)

validation_set = Dataset_RNN(Xval,Yval)
val_loader = data.DataLoader(validation_set,  batch_size=cf_a.batch_size, shuffle=True)

"""
######################## Instantiate Architecture ########################
"""
#myGeneralVBModel = RNN_names_classifier(cf_a, prior)
myGeneralVBModel = RNN_names_classifier_fullVB(cf_a, prior)
myGeneralVBModel.set_languages(all_categories)

#print (myGeneralVBModel.predict_language(Xtrain))

# Set the model in training mode for the forward pass. self.training = True
# This is for Dropout and VI
myGeneralVBModel.train()  

if (load_previous_state):
    files_path = ul.get_allPaths(folder_model)
    if (len(files_path) >0):
        # Load the latest params !
        files_path = sorted(files_path, key = ul.cmp_to_key(ul.filenames_comp_model_param))
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
        print ("Doing epoch: %i"%(i+1))
        if (i == 0):
            ## First iteration we save the initialized values
            myGeneralVBModel.eval()
            myGeneralVBModel.set_posterior_mean(True)
            tr_data_loss.append(myGeneralVBModel.get_data_loss(Xtrain, Ytrain).item())
            val_data_loss.append(myGeneralVBModel.get_data_loss(Xtrain, Ytrain).item())
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
        
        """
        We cannot use the train_loader anymore because the inputs have different length !! 
        We use normal batchloader from CV 
        
        """
        
        order_samples = np.random.permutation(Ntrain)
        Nbatches = int(Ntrain/cf_a.batch_size)
        
        
        def randomChoice(l):
            return np.random.randint(0, l- 1)
        
        def get_random_par_batch_index(Xtrain, Ytrain, batch_size = 20):
            indexes = []
            for i in range(batch_size):
                class_i = randomChoice(cf_a.D_out)
                elements =  np.where(Y_data_tr2== class_i)[0]
                sample_i = elements[randomChoice(elements.size)]
                indexes.append(sample_i)
            return indexes
        
        for b_i in range(Nbatches):
            
            # Since this is an unbalancef problem we can instead draw random samples equally from the different classes
            if(0):
                local_batch = [Xtrain[s_i] for s_i in order_samples[b_i*cf_a.batch_size: (b_i+1)*cf_a.batch_size]]
                local_labels = Ytrain[order_samples[b_i*cf_a.batch_size: (b_i+1)*cf_a.batch_size]]
            else:
                indexes = get_random_par_batch_index(Xtrain, Ytrain, batch_size = cf_a.batch_size)
                local_batch = [Xtrain[s_i] for s_i in indexes]
                local_labels = Ytrain[indexes]
                
                

            
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
        
        if(save_model_per_epochs):
            myGeneralVBModel.save(folder_model + "model_parameters_epoch:%i.pk"%i)


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
    
    
