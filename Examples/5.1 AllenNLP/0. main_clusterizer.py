"""
Python code that will call several times the training program several times
for different possible hyperparamters.

############### THIS SHOULD BE THE ONLY FILE MODIFIED ON THE SERVER ##########
"""
import subprocess
import numpy as np

################## CONFIG OPTIONS ###############

trials_system = False 
Bayesian_output = False

file_path = 'python "./3.main_BiDAF_train.py"'
args = " results_root_folder=../Baseline_model/"
args += " datareader_lazy=True"
args += " save_weights_at_each_epoch=True"
args += " num_epochs=10"

## Crossvalidation of training parameters
args += " ELMO_droput=0.2"
args += " optimizer_params={'lr':1e-3,'betas':[0.9,0.9]}"

## Bayesian inputs 
if (Bayesian_output):

    args += " VB_span_start_predictor_linear=True"
    args += " VB_span_start_predictor_linear_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(0.4), np.log(1))
    args += " VB_span_end_predictor_linear=True"
    args += " VB_span_end_predictor_linear_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(0.4), np.log(1))
    
    args += " VB_span_start_predictor_linear=True"
    args += " VB_span_end_predictor_linear_prior={'pi':%.20f,'log_sigma1':%.20f,'log_sigma2':%.20f}"\
            %(0.5, np.log(0.4), np.log(1))
    
    
if (trials_system):
    args += " instances_per_epoch_train=500"   
    args += " instances_per_epoch_validation=500"
    
Number_executions = 1
for i in range(Number_executions):
    print ("Executing launch (%i/%i) "%(i+1, Number_executions))
    subprocess.run(file_path + args, shell=True)
