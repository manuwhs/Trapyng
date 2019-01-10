"""
This file loads:
    - It loads all the training information (pickle) from a specified folder with logic:
        It scans every subfolder and gets the filepath of the pickle.
    - Then it sorts them in a list by the parameters from which they differ
    
Once it has all this information it plots:
    - 
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
from bidaf_utils import init_training_logger, config_bidaf, print_conf_params,fill_evaluation_data,question_types
from bidaf_model import BidirectionalAttentionFlow_1
import pyTorch_utils as pytut
import pickle_lib as pkl
import utilities_lib as ul
from graph_lib import gl
import matplotlib.pyplot as plt
import squad_plotting as spl
from QA_ensemble import QA_ensemble
import utilities_lib as ul
plt.close("all")
from other_func_bidaf import *
"""
#################### EXECUTION FLAGS ####################
"""

## Information to load the experiments
source_path = "../CV_baseline_1H_ownPC/" 
## Folder
folder_images = "../pics/Thesis/CV_baseline/"
images_prefix = "Baseline_reduced" # We add this prefix to the image names to avoid collision
ul.create_folder_if_needed(folder_images)

Load_models = True
if (Load_models):
    
    pickle_results_path = pytut.get_all_pickles_training(source_path)
    
    Nmodels = len(pickle_results_path)
    
    """
    #################### Initialize random generators and device ####################
    """
    dtype = torch.float
    device = pytut.get_device_name(cuda_index = 0)
    
    """
    ##################################################################
    LOAD THE CONFIGURATION FILES
    ##################################################################
    """


    cf_a_list = []
    training_logger_list = []
    for i in range(Nmodels):
        [cf_a,training_logger] = pkl.load_pickle(pickle_results_path[i])
        ## Set data and device parameters
        cf_a.dtype = dtype  # Variable types
        cf_a.device = device
        cf_a_list.append(cf_a)
        training_logger_list.append(training_logger)
    
    """
    GET THE MODELS WITH BEST PARAMETERS !!
    """
    Nbest = 5
    models_performance = []
    for i in range(Nmodels):
    #    models_performance.append(training_logger_list[i]["validation"]["em"][-1])
        models_performance.append(training_logger_list[i]["validation"]["f1"][-1])
    #    models_performance.append(np.max(training_logger_list[i]["validation"]["em"]))
    #    models_performance.append(np.max(training_logger_list[i]["validation"]["f1"]))
    
    sorted_performance, order_performance = ul.sort_and_get_order(models_performance)
    
    for i in range(Nbest):
        print ("Performance: ", models_performance[order_performance[i]])
        print (cf_a_list[order_performance[i]].command_line_args)


List_analyses = ["Learning Rate", "ELMO DOr", "Batch Size", 
                 "Layers DOr", "Lazy Loading Max","LSTMs hidden size", "Betas",
                 "Run"]
List_f_analyses = [return_lr,return_ELMoDOr,return_batch_size,
                   return_layers_dropout, return_lazy_loading_size, return_hidden_size,
                   return_betas,return_initialization_index]

#Standard_values_alyses = [0.001,0.2,40,
#                          0.2,10000,100,
#                          0.9, "whatever"]
Standard_values_alyses = [0.001,0.2,30,
                          0.2,10000,100,
                          0.9, "whatever"]
"""
Get the data from the training_logger for the charts for Batch info
"""
N_analyses = len(List_analyses)
for an_i in range(N_analyses):
    
    gl.init_figure();
    ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
    
    ## Subselect models to plot !!!
    ## Models with Lazy loading True, 35 epochs
    
    Selected_training_logger_list = []
    Selected_cf_a_list = []
    for i in range (Nmodels):
        training_logger = training_logger_list[i]
        cf_a = cf_a_list[i]
        
        condition_analysis = True
        # Substract one because the last one is the iterations one.
        for an_j in range(N_analyses -1):
            if(an_j != an_i):
                # MAnual fix for Dropout
                if ((an_i == 3)&(an_j == 1)): # The 2 dropouts
                    pass
                else:
                    condition_analysis &= List_f_analyses[an_j](cf_a) == Standard_values_alyses[an_j]
#        condition_analysis &= len(training_logger["train"]["em"]) == 15
        
        if (condition_analysis):
               Selected_training_logger_list.append(training_logger)
               Selected_cf_a_list.append(cf_a)
    
    ########## Dont accept duplicates (if any) and order by the key

    Selected_cf_a_list,Selected_training_logger_list =  \
    order_remove_duplicates(Selected_cf_a_list,Selected_training_logger_list, List_f_analyses[an_i])
    Nselected_models = len(Selected_cf_a_list)
    
    return_initialization_index.initialization_aux = 0;
    
    for i in range(Nselected_models):
        # Only show 10 for the model
        training_logger = Selected_training_logger_list[i]
        if(an_i == N_analyses -1):
            if (training_logger["train"]["f1"][-1] < 0.85):
                continue;
        cf_a = Selected_cf_a_list[i]
        # Final epoch accuracy
        em_train = 100*np.array(training_logger["train"]["em"]) # Remember this is cumulative of all the training
        f1_train = 100*np.array(training_logger["train"]["f1"]) # Remember this is cumulative of all the training
        
        em_validation = 100*np.array(training_logger["validation"]["em"])
        f1_validation = 100*np.array(training_logger["validation"]["f1"])
    
        max_color = 0.8 # The whitest
#        tr_color = [(max_color)*(float(i)/Nselected_models),(max_color)*(float(i)/Nselected_models),1]
        tr_color = [1- (max_color)*(float(1+i)/(Nselected_models))]*3
        
        # Value:
        value = str( List_f_analyses[an_i](cf_a))
        
        # F1 values train
        x_values = 1 + np.array(range(len(f1_train))) 
        gl.plot(x_values, f1_train, ax = ax1, labels = ["F1 score TR","epoch", "F1 score"],color = tr_color,
                legend = [List_analyses[an_i] +": " + value])
        
        # F1 values val
        x_values = 1 + np.array(range(len(f1_train)))  
        gl.plot(x_values, f1_validation, ax = ax2, labels = ["F1 score DEV","epoch", "F1 score"],color = tr_color,
                legend = [List_analyses[an_i] +": " + value])
        
            # start span and end span values
    
    gl.set_fontSizes(ax = [ax1,ax2], title = 18, xlabel = 15, ylabel = 15, 
                      legend = 12, xticks = 12, yticks = 12)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, 
                       wspace=.20, hspace=0.10)
    
    gl.savefig(folder_images + images_prefix + "CV_baseline_"+List_analyses[an_i].replace(" ","_") + ".png",  
               dpi = 100, sizeInches = [14, 5], close = False, bbox_inches = "tight") 

    if (an_i == N_analyses-1):
        # Compute the std of the final F1.
        last_values_tr = []
        last_values_val = []
        for i in range(Nselected_models):
            training_logger = Selected_training_logger_list[i]
            if (training_logger["train"]["f1"][-1] > 0.85):
                last_values_tr.append(100*training_logger["train"]["f1"][-1])
                last_values_val.append(100*training_logger["validation"]["f1"][-1])
        print("Baseline models")
        print ("TR: ", np.mean(last_values_tr), np.std(last_values_tr))
        print ("VAL: ", np.mean(last_values_val), np.std(last_values_val))
    """
    #################### PLOT THE EVOLUTION OF the LAZY LOADING 
    """
    if (an_i ==4 ):
        gl.init_figure();
        axes_list = []
        
        for i in range(Nselected_models):
            ax_ref1 = None
            ax_ref2 = None
            if (i > 0):
                ax_ref1 = axes_list[0]
                ax_ref2 = axes_list[1]
            ax1 = gl.subplot2grid((Nselected_models,2), (i,0), rowspan=1, colspan=1, sharey = ax_ref1, sharex = ax_ref1)
            ax2 = gl.subplot2grid((Nselected_models,2), (i,1), rowspan=1, colspan=1, sharex = ax1, sharey = ax_ref2)
            axes_list.append(ax1); axes_list.append(ax2)
            
        for i in range(Nselected_models):
            
            training_logger = Selected_training_logger_list[i]
            cf_a = Selected_cf_a_list[i]
            
            data_loss_batch = training_logger["train"]["loss_batch"]
            em_train_batches = training_logger["train"]["em_batch"]
            f1_train_batches = training_logger["train"]["f1_batch"]
            
            ax1 = axes_list[i*2]
            ax2 = axes_list[i*2 +1]
            x_values = np.array(range(len(data_loss_batch))) * cf_a.logging_lossesMetrics_batch_frequency /1000.0
            
            # The E
            if i == 0:
                title = "Instant batches training loss"
            else:
                title = ""
            gl.plot(x_values ,data_loss_batch, ax = ax1, color = tr_color,
                    labels = [title,r"Batch $(x10^3)$","Loss"], legend = ["Samples Lazy: " +str( return_lazy_loading_size(cf_a))])
           
            ## Cumulative EM and F1
            x_values = np.array(range(len(em_train_batches))) * cf_a.logging_lossesMetrics_batch_frequency /1000.0
            gl.plot(x_values, em_train_batches, ax = ax2,
                    labels = [title,r"Batch $(x10^3)$", "score"],
                    legend = ["EM" + "Samples Lazy: " +str( return_lazy_loading_size(cf_a))], color = tr_color)
            gl.plot(x_values, f1_train_batches,  legend = ["F1"+"Samples Lazy: " +str( return_lazy_loading_size(cf_a))], color = tr_color, ls = "--")
    
        gl.set_fontSizes(ax = axes_list, title = 18, xlabel = 15, ylabel = 18, 
                          legend = 12, xticks = 14, yticks = 14)
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0)
        
        gl.savefig(folder_images + images_prefix + "CV_baseline_samples_lazy_dampening.png",  
                   dpi = 100, sizeInches = [16, 2*Nselected_models], close = False, bbox_inches = "tight") 
        