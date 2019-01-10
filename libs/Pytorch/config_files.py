# -*- coding: utf-8 -*-
"""
"""
import torch
import numpy as np
import torch.nn as nn

class cf_a_Regression_1D():  # configuration cf_a
    task_type = "regression"
    ## Data dependent:
    D_in = None     # Dimensionality of input features
    D_out = None   # Dimensionality of output features

    ## DEVICES
    dtype = None  # Variable types
    device = None
    
    ## Architecture 
    H = 20      # Number of hidden neurons
    ### Nonlinearity
    activation_func = torch.tanh #   torch.cos  torch.clamp tanh
    
    ### Training 
    loss_func = nn.MSELoss()
    Nepochs = 100       # Number of epochs
    batch_size_train = 3     # Batch size

    ## The optimizer could be anything 
#    optimizer_type = "SGD"
#    optimizer_params = {"lr":1e-2, "weight_decay":0.01}
    optimizer_type = "Adam"   
    optimizer_params = {"lr":1e-2, "betas":(0.9, 0.9),"weight_decay":0.00}
#    
    # Outp
    dop = 0.0 # Dropout p 
    """
    ################ BAYESIAN PARAMS ###################3
    """
    # Weight applied to tbe KL 
    eta_KL = 1
    Nsamples_train = None # We need this value together with batch_size 
                            # for computing the loss of the ELBO

    input_layer_prior  = \
    {"pi":0.5, "log_sigma1":np.log(1), "log_sigma2":np.log(1)}
    output_layer_prior  = input_layer_prior
                
"""
Configuration for trials of the training algorithm !
"""
class cf_classification_1():  # configuration cf_a
    ## Data dependent:
    task_type = "classification"
    D_in = None     # Dimensionality of input features
    D_out = None   # Dimensionality of output features

    ## DEVICES
    dtype = None  # Variable types
    device = None
    
    ## Architecture 
    H = 10      # Number of hidden neurons
    ### Nonlinearity
    activation_func = torch.tanh #   torch.cos  torch.clamp tanh
    
    ### Training 
    loss_func = nn.CrossEntropyLoss()
    Nepochs = 100       # Number of epochs
    batch_size_train= 3     # Batch size
    
    # Outp
    dop = 0.0 # Dropout p 

    ## The optimizer could be anything 
    optimizer_type = "SGD"
    optimizer_params = {"lr":1e-2, "weight_decay":0.0}
    optimizer_type = "Adam"   
    optimizer_params = {"lr":1e-3, "betas":(0.9, 0.9),"weight_decay":0.00}
    """
    ################ BAYESIAN PARAMS ###################3
    """
    # Weight applied to tbe KL 
    eta_KL = 0
    Nsamples_train = None # We need this value together with batch_size 
                            # for computing the loss of the ELBO

    input_layer_prior  = \
    {"pi":0.5, "log_sigma1":np.log(0.4), "log_sigma2":np.log(0.9)}
    output_layer_prior  = input_layer_prior
                
    
"""
Configuration for trials of the training algorithm !
"""
class cf_RNN_1():  # configuration cf_a
    ## Data dependent:
    task_type = "regression"
    D_in = None     # Dimensionality of input features
    D_out = None   # Dimensionality of output features

    ## DEVICES
    dtype = None  # Variable types
    device = None
    HS = 5
    ## Architecture 
    H = 5      # Number of hidden neurons
    ### Nonlinearity
    activation_func = torch.tanh #   torch.cos  torch.clamp tanh
    
    ### Training 
    loss_func = None
    Nepochs = 200       # Number of epochs
    batch_size = 50     # Batch size
    
    ## The optimizer could be anything 
    optimizer = None
    optimizer_params = None
    lr = 0.01
    
    # Outp
    dop = 0.0 # Dropout p 
    
    
    """
Configuration for trials of the training algorithm !
"""
class cf_RNN_2():  # configuration cf_a
    ## Data dependent:
    task_type = "classification"
    D_in = None     # Dimensionality of input features
    D_out = None   # Dimensionality of output features

    ## DEVICES
    dtype = None  # Variable types
    device = None
    HS = 20
    ## Architecture 
    H = 5      # Number of hidden neurons
    ### Nonlinearity
    activation_func = torch.tanh #   torch.cos  torch.clamp tanh
    
    ### Training 
    loss_func = None
    Nepochs = 200       # Number of epochs
    batch_size = 50     # Batch size
    
    ## The optimizer could be anything 
    optimizer = None
    optimizer_params = None
    lr = 0.01
    
    # Outp
    dop = 0.0 # Dropout p 
    