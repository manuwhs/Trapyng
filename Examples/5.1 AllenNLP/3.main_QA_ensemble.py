"""
This file loads:
    - It instantiates the ELMO + BiDAF architecture.
    - It loads from disk the parameters
    - It loads the dataset
    - It loads the training information (pickle)

Once it has all this information it plots:
    - Your mom

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

plt.close("all")
"""
#################### EXECUTION FLAGS ####################
"""
Experiments_load_dataset_from_disk = 1
Experiments_instantiate_model = 1
Experiments_generate_results_data = 1
## File execution files
analyze_example_query = 1
plot_training_results_pickle = 0
train_validation_dataset = False
# Options
Evaluate_Model_Results = 1

## Information to load the experiments
images_prefix = "QA_"
source_path = "../CV_baseline/results/bidaf/" 
list_model_ids = ["2018-11-29 18_50_20.230947","2018-11-29 19_01_23.280821","2018-11-29 19_02_08.270575",
 "2018-11-29 19_21_02.604106","2018-11-30 03_42_17.215512","2018-11-30 03_42_27.543800",
 "2018-11-30 04_20_07.469757"]
Nmodels = len(list_model_ids)
for i in range(Nmodels):
    list_model_ids[i] = list_model_ids[i].replace("_",":")
list_models_epoch_i = [-1]*Nmodels

pickle_results_path_list = []
model_file_path_list = []
Nmodels = len(list_model_ids)


#Nmodels = 3
for i in range(Nmodels):
    pickle_results_path,model_file_path = pytut.get_models_paths(list_model_ids[i], 
                                                    list_models_epoch_i[i],source_path = source_path)
    pickle_results_path_list.append(pickle_results_path)
    model_file_path_list.append(model_file_path)
    

"""
##################################################################
LOAD THE CONFIGURATION FILES
##################################################################
"""
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)
cf_a_list = []
training_logger_list = []
for i in range(Nmodels):
    [cf_a,training_logger] = pkl.load_pickle(pickle_results_path_list[i])
    ## Set data and device parameters
    cf_a.dtype = dtype  # Variable types
    cf_a.device = device
    cf_a.datareader_lazy = True # Force lazyness for RAM optimization
    cf_a.batch_size_train = 30
    cf_a.batch_size_validation = 30
    cf_a.max_instances_in_memory = 1000
    ## Fix backwards compatibility
    cf_a.phrase_layer_hidden_size = cf_a.modeling_span_end_hidden_size
    cf_a.phrase_layer_hidden_size = cf_a.modeling_span_end_hidden_size
    
    cf_a_list.append(cf_a)
    training_logger_list.append(training_logger)
    

"""
##################################################################
############ INSTANTIATE DATAREADER AND LOAD DATASET ############
##################################################################
"""
vocab = Vocabulary()
if(Experiments_load_dataset_from_disk):
    squad_reader, num_batches, train_iterable, num_batches_validation, validation_iterable = \
        load_SQUAD1_dataset(cf_a, vocab)
    
    
"""
######################################################################
################# INSTANTIATE THE INDIVIDUAL MODELS ###################
"""
models_list = []
if (Experiments_instantiate_model):
    for i in range(Nmodels):
        print ("Initializing Model architecture")
        ## SHARED ELMO BETWEEN ALL OF THEM !! 
        preloaded_elmo = None
        if(i > 0):
           preloaded_elmo = models_list[0]._text_field_embedder
        
        cf_a = cf_a_list[i]
        model = BidirectionalAttentionFlow_1(vocab, cf_a,preloaded_elmo)
        print("Loading previous model")
        model.load_state_dict(torch.load(model_file_path_list[i]))
        models_list.append(model)
        ## Dont move all the models to device !! They dont fit !!
        ## Move them one by one in the forward pass.
#        model.to(device = device, dtype = dtype)
        
    model = QA_ensemble(submodels = models_list, load_models = False)

"""
##################################################################
############ INSTANTIATE DATAREADER AND LOAD DATASET ############
##################################################################
"""
vocab = Vocabulary()
if(Experiments_load_dataset_from_disk):
    squad_reader, num_batches, train_iterable, num_batches_validation, validation_iterable = \
        load_SQUAD1_dataset(cf_a, vocab)
        
"""
#####################################################################
################# OBTAIN THE DATA FROM THE DATASET ###################
"""

#def fill_DataSet_statistics_runner(model, dataset_iterable, num_batches):
if (train_validation_dataset):
    images_prefix += "train_"
    dataset_iterable = train_iterable
    num_batches = num_batches
else:
    images_prefix += "validation_"
    dataset_iterable = validation_iterable
    num_batches = num_batches_validation
    
if (Experiments_generate_results_data):
    DataSet_statistics = fill_evaluation_data(model,device, dataset_iterable,num_batches, Evaluate_Model_Results)

EM = np.mean(DataSet_statistics["em"])
F1 = np.mean(DataSet_statistics["f1"])

print ("EM: ", EM, ", F1: ",F1)
