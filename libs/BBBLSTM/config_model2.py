"""
Configuration files for model 2
"""


class init_config(object):
    name = "truncated_normal"
    std = 0.1
    
class loss_config(object):
    """
    Configuration parameters for the loss function
    """
    name = "Hubber"   # "MSE"
    delta = 1.0
    
class train_config(object):
    """
    Configuration parameters for the training algorithm
    """
    name = "GradientDescent"           # "GradientDescent"    "Adam"    "RMSProp"
    lr = 0.01              # Learning Rate. 
    
    ## Adam parameters 
    Adam_beta1 = 0.9;
    Adam_beta2 = 0.999;
    Adam_epsilon = 1e-08
    
    ## RMSProp
    RMSProp_decay = 0.9
    RMSProp_momentum = 0.0
    RMSProp_epsilon = 1e-10
    RMSProp_use_locking = False
    RMSProp_centered=False
    
    saver = True;
    variables_path = None;
    ##################################################
    
    
class model_config(object):
    """
    Model config for the architecture and dimensions of the network.
    This is very dependent from model to model.
    """
    
#    num_hidden = [10,14]  # Number of hidden neurons in each layer
#    non_linearity = ["tanh", "linear"]   

    num_hidden1 = 1
    num_hidden2 = 1
    non_linearity1 = "tanh"
    non_linearity2 = "tanh"
    
    input_dim = None  # Dimensionality of the input depends on the problem
    output_dim = None # Dimensionality of the output depends on the problem
    
    output_func = "linear" # Operation to the output layer 