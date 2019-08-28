
"""
This file trains the ELMO + BiDAF architecture.
    - All the information needed to characterize the hyperparameters of the model is in the loaded
    - Parameters can be overwritten when calling the program using: parameter_name=value
      The parser will try to convert the number to int or float if necessary by checking the default value type
      zmart hum ?
    - If the folders where to place the results have not been created, they will
"""

###### pyTorch modules ####
import torch
from allennlp.data import Vocabulary
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
import datetime as dt
import copy

## Move directory to the source code folder and import path of modules
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

## Import own libraries
from squad1_reader import Squad1Reader, load_SQUAD1_dataset
from bidaf_model import BidirectionalAttentionFlow_1
from bidaf_utils import init_training_logger, config_bidaf, parse_args_architecture, email_config, send_email, send_error_email
import pyTorch_utils as pytut
import utilities_lib as ul
import pickle_lib as pkl
from graph_lib import gl
import time
"""
#################### PARSE PARAMETERS ####################
Maybe we were given parameters from the command line, mainly to be able to 
execute different parameters of the architecture.

To use it in Spyder:
    runfile('3.main_BiDAF_train.py', wdir='/5.1 AllenNLP',args='one two three')

Example of parameters:
    num_highway_layers=2 
    VB_span_start_predictor_linear=False
    VB_span_start_predictor_linear_prior={'pi':0.5,'log_sigma1':-2.302,'log_sigma2':-0.916}
    optimizer_type='Adam'
    optimizer_params={'lr':1e-3,'betas':(0.9,0.9)}
"""
argv_params = copy.deepcopy(sys.argv)
argv_params.pop(0)
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', sys.argv)
cf_a = config_bidaf()
cf_a = parse_args_architecture(argv_params, cf_a)

"""
#################### EXECUTION FLAGS ####################
- These flags are just meant to disable certain execution parts to perform experiments,
not having to wait for those parts to execute.
"""

Experiments_load_dataset_from_disk = 1  # Load the dataset from disk and comput the Tokens and Indexing
Experiments_instantiate_model = 1
Experiments_train_model = 1
Experiments_Bayesian_checking = 0

if (0): ## Experiments Bayesian
    cf_a.max_instances_in_memory = 1000
    cf_a.batch_size_train = 20
    cf_a.batch_size_validation = 20
    
    cf_a.instances_per_epoch_train = 100 #  87599 1000
    cf_a.instances_per_epoch_validation = 100 # 10570 1000
    
    cf_a.VB_similarity_function = True
    cf_a.VB_highway_layers = True
    cf_a.VB_span_start_predictor_linear = True
    cf_a.VB_span_end_predictor_linear = True 
    
"""
############## Create folder directories ################
Informatrion about where to save the partial results
"""
time_now = dt.datetime.now()
root_folder = cf_a.results_root_folder
 
pickle_results_path = root_folder + "results/bidaf/" + str(time_now) + "/training_logger/"
mode_file_path = root_folder + "results/bidaf/" + str(time_now) + "/models/"
ul.create_folder_if_needed(pickle_results_path)
ul.create_folder_if_needed(mode_file_path)

"""
#################### Initialize random generators ####################
This will help make results more reproducible but there are at least these sources of uncertainty:
    - Different GPU's might have different random generators.
    - Also the generators of GPU and CPU are different
    - ELMO estates are not reinitialized per batch, so we wil always get somewhat different
       results every time we execute them. It is recommend it to warm it up after loading
       by propagating a few samples first.
    - The CUDA operations are not deterministic 
"""
random_seed = time_now.microsecond
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

"""
#################### LOAD CONFIGURATION FILE ######################
We set the device and dtype externally as it depends on the machine we are using
"""
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)

## Set data and device parameters
cf_a.dtype = dtype  
cf_a.device = device

# We do not need vocabulary in this case, all possible chars ?
vocab = Vocabulary()

"""
##################################################################
############ INSTANTIATE DATAREADER AND LOAD DATASET ############
##################################################################
"""
if(Experiments_load_dataset_from_disk):
    squad_reader, num_batches, train_iterable, num_batches_validation, validation_iterable = \
        load_SQUAD1_dataset(cf_a, vocab)

"""
########################################################
################# INSTANTIATE THE MODEL ###################
"""
Ntrials_CUDA = 100
if (Experiments_instantiate_model):
    
    model = BidirectionalAttentionFlow_1(vocab, cf_a)
    
    trial_index = 0
    while (1):
        try:
            model.to(device = device, dtype = dtype)
            break;
        except RuntimeError as er:
            print (er.args)
            torch.cuda.empty_cache()
            time.sleep(5)
            torch.cuda.empty_cache()
            trial_index += 1
            if (trial_index == Ntrials_CUDA):
                print ("Too many failed trials to allocate in memory")
                send_error_email(str(er.args))
                sys.exit(0)
    

"""
########################################################
################# BAYESIAN EXPERIMENTS AND CHECKING ###################
"""
if (Experiments_Bayesian_checking):
    
    if(model.cf_a.VB_span_start_predictor_linear == True):
        weights_mean = model._span_start_predictor_linear.mu_weight.detach().cpu().numpy().flatten()
        bias_mean = model._span_start_predictor_linear.mu_bias.detach().cpu().numpy().flatten()
    else:
        weights_mean = model._span_start_predictor_linear.weight.detach().cpu().numpy().flatten()
        bias_mean = model._span_start_predictor_linear.bias.detach().cpu().numpy().flatten()
    
    gl.init_figure()
    gl.scatter(np.zeros(weights_mean.size), weights_mean,alpha = 0.2)

"""
#############################################################
####################### TRAIN MATE ##########################
#############################################################
"""

if (Experiments_train_model):
    # Data structure that contains the information about training 
    training_logger = init_training_logger()
    num_epochs = cf_a.num_epochs
    
    
    for i in range(num_epochs):
        start_epoch_time = dt.datetime.now()
        
        if (i == 0):
            pass
        
        print ("Starting epoch: %i"%(i+1))
        
        model.train()
        model.set_posterior_mean(False)
        for j in range(num_batches):
            
            """
            In order to avoid that a collision in memory destroys the entire training,
            we catch the CUDA lack of memory error, wait for a second, and retry up to 10 times,
            then give up
            """
            trial_index = 0
            tensor_dict = next(train_iterable)          # Get the batch cuda
            
            while (1):
                try:
                    tensor_dict = pytut.move_to_device(tensor_dict, device) ## Move the tensor to cuda
                    output_batch = model.train_batch(**tensor_dict)
                    break;
                except RuntimeError as er:
                    print ("Error during training batch. Waiting a minute")
                    print (er.args)
                    torch.cuda.empty_cache()
                    time.sleep(5)
                    torch.cuda.empty_cache()
                    trial_index += 1
   
                    if (trial_index == Ntrials_CUDA):
                        print ("Too many failed trials to allocate in memory")
                        send_error_email("Ran out of cuda memory")
                        sys.exit(0)
                        
                        
            if ((j+1) % cf_a.logging_lossesMetrics_batch_frequency == 0):
                ## Logging the data from the batches

                model.fill_batch_training_information(training_logger, output_batch)
#                print ("Batch %i/%i. Losses: Train(%.2f)"% \
#                       (j+1,num_batches, training_logger["train"]["loss_batch"][-1]))

                
            if ((j+1) % cf_a.logging_VB_weights_batch_frequency == 0):
                pass
            
            ## Memmory management !
            if (cf_a.force_free_batch_memory):
                del tensor_dict["question"]; del tensor_dict["passage"]
                del tensor_dict
                del output_batch
                torch.cuda.empty_cache()
            if (cf_a.force_call_garbage_collector):
                gc.collect()
                        
        time_now = dt.datetime.now()
        epoch_duration = time_now - start_epoch_time
        time_prev = time_now
        print ("Training Epoch duration: ", epoch_duration)
        training_logger["time"]["training"].append(epoch_duration)
        
        model.fill_epoch_training_information(training_logger,device,
                                        validation_iterable, num_batches_validation)
        time_now = dt.datetime.now()
        validation_duration = time_now - time_prev
        time_prev = time_now
        print ("Validation duration: ", validation_duration)
        training_logger["time"]["validation"].append(validation_duration)
        
        ### Save the results to disk !!!
        cf_a.mode_file_path = mode_file_path + str(i+1) + ".prm"
        cf_a.pickle_results_path = pickle_results_path + str(i+1) + ".pkl"
        pkl.store_pickle(pickle_results_path + str(i+1) + ".pkl", [cf_a,training_logger])
        
        if (cf_a.save_weights_at_each_epoch):
            torch.save(model.state_dict(), mode_file_path + str(i+1) + ".prm")
        
        
        # Remove the previous pickle file
        if(i > 0):
            os.remove(pickle_results_path + str(i) + ".pkl")
        if (cf_a.save_weights_at_each_epoch):
            if ((i > 0) and (cf_a.save_only_last_weights)):  # Erase the previous one 
                os.remove(mode_file_path + str(i) + ".prm")
            
        time_now = dt.datetime.now()
        saving_duration = time_now - time_prev
        time_prev = time_now
         
        training_logger["time"]["saving_parameters"].append(saving_duration)
        print ("Saving duration: ", saving_duration)
        duration_total_epoch = time_now - start_epoch_time
        training_logger["time"]["epoch"].append(epoch_duration)
    """
    ################### SEND EMAIL IN THE END WITH MAIN RESULTS #####################
    """
    send_email(email_config, training_logger, cf_a)
        
    