
"""
This file evaluates the result of a single trained model. 
Plotting its individual results from the training_logger
This file loads:
    - It instantiates the ELMO + BiDAF architecture.
    - It loads from disk the parameters
    - It loads the dataset
    - It loads the training information (pickle)

Once it has all this information it plots:
    - Plots of the training of the loaded model from the pickled file

"""
###### pyTorch modules ####
import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
"""
Loses and accuracies
"""
from allennlp.data.iterators import BucketIterator
import torch.optim as optim

# Standard modules
import numpy as np
import gc
import os
import sys

## Move directory to the source code folder and import path of modules
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

## Import own libraries
from squad1_reader import Squad1Reader, load_SQUAD1_dataset
from bidaf_utils import print_conf_params
from bidaf_model import BidirectionalAttentionFlow_1
import pyTorch_utils as pytut
import pickle_lib as pkl
import utilities_lib as ul
from graph_lib import gl
import matplotlib.pyplot as plt
import squad_plotting as spl

plt.close("all")
"""
#################### EXECUTION FLAGS ####################
"""

Epoch_related = 1
Batch_related = 1
Time_related = 1

folder_images = "../pics/Thesis/Eval_model_trainlogger/"
images_prefix = "baseline2" # We add this prefix to the image names to avoid collision
ul.create_folder_if_needed(folder_images)
"""
#################### MODEL TO LOAD ####################
Information to load the experiments
"""

source_path = "../CV_baseline/results/bidaf/"
model_id = "2018-11-29 18:50:20.230947" # "2018-10-12 08:55:20.009008" 
#  "2018-10-10 21:16:33.652684"
 # "2018-09-30 00:29:50.035864" #  "2018-09-26 20:41:06.955179"  # "2018-09-27 08:36:18.902846"# "2018-09-26 20:41:06.955179" 
 
pickle_results_path,model_file_path = pytut.get_models_paths(model_id, 
                                epoch_i = -1,source_path = source_path)


#### Different options !!
#source_path = "../Baseline_model/" 
#pickle_results_path = pytut.get_all_pickles_training(source_path)[0]
#
#source_path = "../CV_baseline/" 
#pickle_results_path = pytut.get_all_pickles_training(source_path)[1]

"""
##################################################################
LOAD THE CONFIGURATION FILE
##################################################################
"""
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)
[cf_a,training_logger] = pkl.load_pickle(pickle_results_path)
cf_a.dtype = dtype  # Variable types
cf_a.device = device

print_conf_params(cf_a)
"""
####################################################################
################# PLOT RESULTS FROM THE TRAINING LOG ###################
"""

if (Epoch_related):
    
    """
    Get the data from the training_logger for the charts for Batch info
    """

    # Final epoch accuracy
    em_train = 100*np.array(training_logger["train"]["em"]) # Remember this is cumulative of all the training
    f1_train = 100*np.array(training_logger["train"]["f1"]) # Remember this is cumulative of all the training
    
    em_validation = 100*np.array(training_logger["validation"]["em"])
    f1_validation = 100*np.array(training_logger["validation"]["f1"]) 

    start_acc_train = 100*np.array(training_logger["train"]["start_acc"]) # Remember this is cumulative of all the training
    end_acc_train = 100*np.array(training_logger["train"]["end_acc"]) # Remember this is cumulative of all the training
    start_acc_validation = 100*np.array(training_logger["validation"]["start_acc"] )
    end_acc_validation = 100*np.array(training_logger["validation"]["end_acc"])


    gl.init_figure();
    ax1 = gl.subplot2grid((1,3), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((1,3), (0,1), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
    ax3 = gl.subplot2grid((1,3), (0,2), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)

    tr_color = "b"
    val_color = "k"
    # EM values
    x_values = 1 + np.array(range(len(em_train))) 
    gl.plot(x_values, em_train, ax = ax1, labels = ["EM score","epoch", "EM score"], color = tr_color,
            legend = ["EM TR"])
    gl.plot(x_values, em_validation,  legend = ["EM DEV"], color = val_color)
    # F1 values
    x_values = 1 +  np.array(range(len(f1_train))) 
    gl.plot(x_values, f1_train, ax = ax2, labels = ["F1 score","epoch", "F1 score"],color = tr_color,
            legend = ["F1 TR"])
    gl.plot(x_values, f1_validation,  legend = ["F1 DEV"],  color = val_color)
    
    # start span and end span values
    x_values = 1 +  np.array(range(len(start_acc_train))) 
    gl.plot(x_values, start_acc_train, ax = ax3, labels = ["Start and End span Acc","epoch", "score"],
            legend = ["start span TR"], color = tr_color)
    gl.plot(x_values, end_acc_train,  legend = ["end span TR"], color = tr_color, ls = "--")
    gl.plot(x_values, start_acc_validation, ax = ax3,
            legend = ["start span DEV"], color = val_color)
    gl.plot(x_values, end_acc_validation,  legend = ["end span DEV"], 
            color = val_color, ls = "--")
    
    
    gl.set_fontSizes(ax = [ax1,ax2,ax3], title = 20, xlabel = 18, ylabel = 18, 
                      legend = 15, xticks = 14, yticks = 14)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.10)
    gl.format_yaxis ( ax = ax3, Nticks = 10);
    gl.format_xaxis ( ax = ax3, Nticks = len(em_train));
    gl.savefig(folder_images + images_prefix + "Accuracies_epoch.png",  
               dpi = 100, sizeInches = [20, 5], close = False, bbox_inches = "tight") 

if (Batch_related):
    """
    #############################################
    Batch plots 
    """

    data_loss_batch = training_logger["train"]["loss_batch"]
    em_train_batches = 100*np.array(training_logger["train"]["em_batch"])
    f1_train_batches = 100*np.array(training_logger["train"]["f1_batch"])
    
    gl.init_figure();
    ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1, sharex = ax1)
    
    x_values = np.array(range(len(data_loss_batch))) * cf_a.logging_lossesMetrics_batch_frequency /1000.0
    
    # The E
    gl.plot(x_values ,data_loss_batch, ax = ax1, color = tr_color,
            labels = ["Instant batches training loss",r"Batch $(x10^3)$","Loss"],
            legend = ["Loss TR"])
   
    ## Cumulative EM and F1
    x_values = np.array(range(len(em_train_batches))) * cf_a.logging_lossesMetrics_batch_frequency /1000.0
    gl.plot(x_values, em_train_batches, ax = ax2,
            labels = ["Cummulative EM and F1",r"Batch $(x10^3)$", "score"],
            legend = ["EM TR"], color = tr_color)
    gl.plot(x_values, f1_train_batches,  legend = ["F1 TR"], color = tr_color, ls = "--")

    gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 15, ylabel = 18, 
                      legend = 12, xticks = 14, yticks = 14)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    
    gl.savefig(folder_images + images_prefix + "Accuracies_batch.png",  
               dpi = 100, sizeInches = [16, 5], close = False, bbox_inches = "tight") 
   
if (Time_related):
    mean_train_epoch = np.mean(training_logger["time"]["epoch"])# The duration of the epoch
    mean_validation_epoch = np.mean(training_logger["time"]["validation"]) # The duration of the epoch
    mean_save_epoch = np.mean(training_logger["time"]["saving_parameters"])

    print ("mean_train_epoch: ", mean_train_epoch )
    print("mean_validation_epoch: ", mean_validation_epoch)
    print ("mean_save_epoch: ", mean_save_epoch)
    

print ("BEST DEV EM: ", training_logger["validation"]["em"][-1])
print ("BEST DEV F1: ", training_logger["validation"]["f1"][-1])