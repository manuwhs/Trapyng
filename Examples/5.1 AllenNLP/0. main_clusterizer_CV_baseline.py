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

import subprocess
import numpy as np
import sys
import copy
#from bidaf_utils import send_error_email

################## CONFIG OPTIONS ###############
file_path = 'python "./3.main_BiDAF_train.py"'
trials_system = False

"""
ATTEMP TO USE THE SCRATCH FOR SAVING THE MODEL PARAMS AND LOADING DATA
"""
## Crossvalidation of training parameters
ELMO_dropout_list = [0.0, 0.1,0.15, 0.2,0.25, 0.3,0.35, 0.4]
Learning_rate = [0.0005, 0.0008,0.001, 0.002, 0.005, 0.01]
Batch_size = [20, 30, 40,50]
Lazy_loading = [5000, 10000, 20000,40000, -1]
betas = [0.85, 0.9, 0.95, 0.99, 0.999]
all_dropouts = [0.0, 0.05, 0.1,0.15, 0.2,0.25,0.30]
hidden_sizes = [50,80,100,120,150,180]

lists = [ELMO_dropout_list,Learning_rate,
         Batch_size,Lazy_loading,betas, all_dropouts,
         hidden_sizes]
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

print ("Cross Validating: ", selected_list)
for i in range(N_runs):
    args = " results_root_folder=../CV_baseline_1Highway/"
    args += " save_weights_at_each_epoch=True" # We are not interested in the final weights
    
    if (trials_system):
        args += " instances_per_epoch_train=500"   
        args += " instances_per_epoch_validation=500"
        args += " batch_size_train=10"
        args += " batch_size_validation=10"
        
    ############ CV of parameters #################3
    if (selected_list_index == 0):
        args += " ELMO_droput=%f"%(selected_list[i])
    elif(selected_list_index == 1):
        args += " optimizer_params={'lr':%f,'betas':[0.9,0.9]}"%(selected_list[i])
    elif(selected_list_index == 2):
        args += " batch_size_train=%i"%(selected_list[i])
        args += " batch_size_validation=%i"%(selected_list[i])
    elif(selected_list_index == 3):
        if (selected_list[i] == -1):
            args += " datareader_lazy=False"
        else:
            args += " max_instances_in_memory=%i"%(selected_list[i])
    elif(selected_list_index == 4):
        args += " optimizer_params={'lr':0.001,'betas':[%f,%f]}"%(selected_list[i],selected_list[i])
    elif(selected_list_index == 5):
        args += " ELMO_droput=%f"%(selected_list[i])
        args += " phrase_layer_dropout=%f"%(selected_list[i])
        args += " modeling_passage_dropout=%f"%(selected_list[i])
        args += " span_end_encoder_dropout=%f"%(selected_list[i])
        args += " spans_output_dropout=%f"%(selected_list[i])
        
    elif(selected_list_index == 6):
        args += " phrase_layer_hidden_size=%i"%(selected_list[i])
        args += " modeling_passage_hidden_size=%i"%(selected_list[i])
        args += " modeling_span_end_hidden_size=%i"%(selected_list[i])
        
    print ("-------Executing launch (%i/%i)----------- "%(i+1, N_runs))
    print (args)
    compleded_subprocess = subprocess.run(file_path + args, shell=True)
#    if(compleded_subprocess.returncode > 0):
#        send_error_email(args)