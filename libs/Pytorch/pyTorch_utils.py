# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter
import utilities_lib as ul

"""
CUDA Tensors
Tensors can be moved onto any device using the .to method.

"""
def get_optimizers(model, cf_a):
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    if (cf_a.optimizer_type == "SGD"):
        optimizer = optim.SGD(parameters, **cf_a.optimizer_params)
    elif(cf_a.optimizer_type == "Adam"):
        optimizer = optim.Adam(parameters, **cf_a.optimizer_params)
    
    return optimizer
    
    
def get_device_name(cuda_index = 0):
    """
    Check the device name and 
    """
    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")          # a CUDA device object
    else:
        device = torch.device("cpu") 
    
    """
    Print available cudas !
    """
    if(torch.cuda.device_count() >0):
        print ("----- CUDA DEVICES AVAILABLE ------------")
        for i in range(torch.cuda.device_count() ):
            print(torch.cuda.get_device_name(i))
#        device = torch.device(torch.cuda.get_device_name(cuda_index)) 
    else: 
        print ("----- NO CUDA DEVICES AVAILABLE ------------")
    
#    device = torch.device("cpu") 
    return device


def move_to_device(obj, device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    if isinstance(obj, torch.Tensor):
        return obj.to(device = device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj

def get_models_paths(model_id, epoch_i = -1, source_path = "../results/bidaf/"):
    """
    Function that returns the path of the files where the data of the model are loaded:
        - pickle_results_path: Where the pickle results with the config_file and training_logger are
        - model_file_path: Where the weights for the model are.
    
    model_id: Identifies the experiment to be loaded
    epoch_id: Identifies the epoch
    """
    pickle_results_folder = source_path +model_id + "/training_logger/"
    model_file_folder = source_path + model_id + "/models/"

    if (epoch_i == -1):
        # Get the latest element form the folders
        files_path = ul.get_allPaths(pickle_results_folder)
        if (len(files_path) >0):
            # Load the latest params !
            files_path = sorted(files_path, key = ul.cmp_to_key(ul.filenames_comp))
            epoch_i = int(files_path[-1].split("/")[-1].split(".")[0])
            print ("Last epoch: ",epoch_i)
    pickle_results_path = pickle_results_folder+ str(epoch_i) + ".pkl"
    model_file_path =model_file_folder + str(epoch_i) + ".prm"

    return pickle_results_path,model_file_path

def get_all_pickles_training(source_path = "../results/bidaf/", include_models = False):
    """
    Function that returns the pickle path of all models in the folder
    """
    pickle_results_path = []
    model_results_path = []
    
    files_path = ul.get_allPaths(source_path)
    if (len(files_path) >0):
        # Load the latest params !
        for i in range(len(files_path)):
            if(files_path[i].split(".")[-1] == "pkl"):
                pickle_results_path.append(files_path[i])
                if (include_models):
                    epoch_i = files_path[i].split("/")[-1].split(".")[0]
                    model_path = "/".join([files_path[i].split("/")[j] \
                    for j in range(len(files_path[i].split("/"))-2)])
                    model_path += "/models/" + str(epoch_i) + ".prm"
                    model_results_path.append(model_path)
    if(include_models):
        return pickle_results_path,model_results_path
    return pickle_results_path

