"""
This file executes the file that trains the BiDAF "./3.main_BiDAF_train.py"
with a set of hyperparameters that can be changed in the code.
The modified parameters are specified as arguments in the calling of the script.

This particular script performs the CV of the hyperparameters of the baseline:
    - Dropout rate of ELMO
    - Learning rate of the adam opmitizer

Each machine should be executed with a parameters 0 - 4
############### THIS SHOULD BE THE ONLY FILE MODIFIED ON THE SERVER ##########
"""
## Move directory to the source code folder and import path of modules

import os
import sys
import subprocess
import numpy as np
import sys
import copy

#from bidaf_utils import send_error_email # Cant be used now due to having to move directory to load
################## CONFIG OPTIONS ###############
file_path = 'python "./3.main_BiDAF_train.py"'
trials_system = False

"""
ATTEMP TO USE THE SCRATCH FOR SAVING THE MODEL PARAMS AND LOADING DATA
"""
## Crossvalidation of training parameters
all_dropouts = [0.0, 0.05, 0.1,0.15, 0.2,0.25,0.30,0.35]
eta_list = [0.0, 0.001, 0.01, 0.1, 1,10]
#eta_list = [0.002, 0.003, 0.005, 0.008]

"""
This one we can select from the command line now separately
"""
sigma2_list= [0.1, 0.3, 0.5, 1.0,2.0,3.0,5.0]

default_dropout = 0.0
default_sigma_1 = 0.5
selected_sigma2 = 0.5;

lists = [all_dropouts,
         eta_list]
"""
CHANGE THIS PARAMETER FROM MACHINE TO MACHINE !!!
"""
selected_list = []
if (len(sys.argv) == 1):
    print ("-------------- WARNING: NO PARAMETERS GIVEN------------------- ")
    
selected_list_index = int(sys.argv[1]) # Selects thelist to CV
# Format of indexes_list: 1,2,3
if (len(sys.argv) > 2):
    indexes_list = sys.argv[2].split(",")    # If we just want a specific se tin the lsit
    print (indexes_list)
    for j in range(len(indexes_list)):
        selected_list.append(lists[selected_list_index][int(indexes_list[j])])
else:
    selected_list = lists[selected_list_index]
    
N_runs = len(selected_list)

if (len(sys.argv) > 3):
    selected_sigma2 = sigma2_list[ int(sys.argv[3]) ]
    print ("Selected log_sigma_2: ", selected_sigma2)
    default_sigma_1 = selected_sigma2
    
if (len(sys.argv) > 4):
    default_dropout = all_dropouts[ int(sys.argv[4]) ]
    print ("Selected default_dropout: ", default_dropout)
    
selected_list_index = int(sys.argv[1]) # Selects thelist to CV

print ("Cross Validating: ", selected_list)
for i in range(N_runs):
    args = " results_root_folder=../CV_bayesian_allbutLSTMS_1Highway_mubias1/"
    args += " save_weights_at_each_epoch=True" # We are not interested in the final weights
    
    ## Lower batch size since we have more parameters now
    args += " batch_size_train=30"
    args += " batch_size_validation=30"
    
    ## We also do not need as many epochs: 
    args += " num_epochs=10"

    ## Set the variational parts
    args += " VB_Linear_projection_ELMO=True"
    args += " VB_highway_layers=True"
    args += " VB_similarity_function=True"
    args += " VB_span_start_predictor_linear=True"
    args += " VB_span_end_predictor_linear=True"
    
    ### Set the dropouts initially to 0
    args += " ELMO_droput=%f"%(default_dropout)
    args += " phrase_layer_dropout=%f"%(default_dropout)
    args += " modeling_passage_dropout=%f"%(default_dropout)
    args += " span_end_encoder_dropout=%f"%(default_dropout)
    args += " spans_output_dropout=%f"%(default_dropout)
    
    if (trials_system):
        args += " instances_per_epoch_train=500"   
        args += " instances_per_epoch_validation=500" 
        args += " num_epochs=1"
        args += " batch_size_train=100"
        args += " batch_size_validation=100"
    
    ############ CV of parameters #################3
    if(selected_list_index == 0):
        args += " ELMO_droput=%f"%(selected_list[i])
        args += " phrase_layer_dropout=%f"%(selected_list[i])
        args += " modeling_passage_dropout=%f"%(selected_list[i])
        args += " span_end_encoder_dropout=%f"%(selected_list[i])
        args += " spans_output_dropout=%f"%(selected_list[i])
        
    elif(selected_list_index == 1):
        args += " eta_KL=%f"%(selected_list[i])
        
    """
    Modify the priors !!
    """
    args += " VB_Linear_projection_ELMO_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(default_sigma_1), np.log(selected_sigma2))
            
    args += " VB_highway_layers_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(default_sigma_1), np.log(selected_sigma2))
    args += " VB_similarity_function_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(default_sigma_1), np.log(selected_sigma2))
            
    args += " VB_span_start_predictor_linear_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(default_sigma_1), np.log(selected_sigma2))
    args += " VB_span_end_predictor_linear_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(default_sigma_1), np.log(selected_sigma2))
            
                
    print ("-------Executing launch (%i/%i)----------- "%(i+1, N_runs))
    print (args)
    compleded_subprocess = subprocess.run(file_path + args, shell=True)
    print (compleded_subprocess)
#    if(compleded_subprocess.returncode > 0):
#        send_error_email(args)