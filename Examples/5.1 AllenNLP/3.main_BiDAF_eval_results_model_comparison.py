
"""

This file loads:
    - It instantiates 2 or more ELMO + BiDAF architecture.
    - It loads from disk the parameters
    - It loads the dataset
    - It loads the training information (pickle)

Once it has all this information it plots:
    - Comparison of models Venn Diagram

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

folder_images = "../pics/Thesis/Ensemble/"
ul.create_folder_if_needed(folder_images)

"""
#################### EXECUTION FLAGS ####################
"""
Experiments_load_dataset_from_disk = 0
Experiments_instantiate_model = 0
Experiments_generate_results_data = 0

# Options
Evaluate_Model_Results = 1

train_validation_dataset = False 
"""
#################### MODELS TO LOAD ####################
Information to load the experiments
"""
 
source_path = "../CV_baseline/results/bidaf/" 
list_model_ids = ["2018-11-29 18_50_20.230947","2018-11-29 19_01_23.280821","2018-11-29 19_02_08.270575",
 "2018-11-29 19_21_02.604106","2018-11-30 03_42_17.215512","2018-11-30 03_42_27.543800",
 "2018-11-30 04_20_07.469757"]
Nmodels = len(list_model_ids)
for i in range(Nmodels):
    list_model_ids[i] = list_model_ids[i].replace("_",":")
list_models_epoch_i = [-1]*Nmodels


Nmodels = 3

pickle_results_path_list = []
model_file_path_list = []


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
    ## Set data and device parameters
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
########################################################
################# INSTANTIATE THE MODEL ###################
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
        
        # We should not move them to the device because they wont fit
        # TODO: Bayesian weights are moved to device by default, we need to change that.
        model.to(device = device, dtype = dtype)
        
#    model = QA_ensemble(submodels = models_list, load_models = False)

"""
#####################################################################
################# OBTAIN THE DATA FROM THE DATASET ###################
"""

#def fill_DataSet_statistics_runner(model, dataset_iterable, num_batches):

if (Experiments_generate_results_data):
    DataSet_statistics_lists = []
    if (train_validation_dataset):
#        images_prefix += "_train"
        dataset_iterable = validation_iterable
        num_batches = num_batches_validation
    else:
#        images_prefix += "_validation"
        dataset_iterable = validation_iterable
        num_batches = num_batches_validation
    
    DataSet_statistics_lists = fill_evaluation_data(models_list,device, dataset_iterable,num_batches, Evaluate_Model_Results)

from matplotlib_venn import venn2,venn3,venn3_circles,venn2_circles
first_model = np.sum(DataSet_statistics_lists[0]["em"])
second_model = np.sum(DataSet_statistics_lists[1]["em"])
intersection = np.sum(np.bitwise_and(DataSet_statistics_lists[0]["em"], DataSet_statistics_lists[1]["em"]))
union = np.sum(np.bitwise_or(DataSet_statistics_lists[0]["em"], DataSet_statistics_lists[1]["em"]))

"""
##################### FOR 2 MODELS ############################3
"""
gl.init_figure()
# First way to call the 2 group Venn diagram:
v = venn2(subsets = (first_model- intersection, second_model- intersection, intersection), 
      set_labels = ('Run 1', 'Run 2'))
c = venn2_circles(subsets = (first_model- intersection, second_model- intersection, intersection), linestyle='dotted',lw = 1.0)
plt.title("Ven diagram of Exact Matches for 2 Runs")
plt.show()

gl.savefig(folder_images + "ven2.png",  dpi = 100, sizeInches = [8, 5], close = False, bbox_inches = "tight") 
   
### 
## Just to make sure the answers are the same, the EM score can he higher than this value. Since
## there are several possible correct answers from which EM is computed.
###
answers_common = 0
for i in range (len(DataSet_statistics_lists[0]["estimated_answer"])):
    if (DataSet_statistics_lists[0]["estimated_answer"][i] == DataSet_statistics_lists[1]["estimated_answer"][i]):
        answers_common+= 1


EM_1 = np.mean(DataSet_statistics_lists[0]["em"])
EM_2 = np.mean(DataSet_statistics_lists[1]["em"])
F1_1 = np.mean(DataSet_statistics_lists[0]["f1"])
F1_2 = np.mean(DataSet_statistics_lists[1]["f1"])

print ("Individual EM: ", np.sum(DataSet_statistics_lists[0]["em"]), np.mean(DataSet_statistics_lists[0]["em"]))

EM_ensemble = np.sum(np.bitwise_or(DataSet_statistics_lists[0]["em"], DataSet_statistics_lists[1]["em"]))
EM_ensemble_percentage = EM_ensemble/len(DataSet_statistics_lists[0]["em"])
print ("Union 2 Runs: ", EM_ensemble, EM_ensemble_percentage)

"""
##################### FOR 3 MODELS ############################3
"""
all_indexes = np.array(range(len(DataSet_statistics_lists[0]["em"])))
set_1 = set(all_indexes[DataSet_statistics_lists[0]["em"]])
set_2 = set(all_indexes[DataSet_statistics_lists[1]["em"]])
set_3 = set(all_indexes[DataSet_statistics_lists[2]["em"]], )

gl.init_figure()
venn3([set_1, set_2, set_3], set_labels =  ('Run 1', 'Run 2', "Run3"))
c = venn3_circles([set_1, set_2, set_3], linestyle='dotted',lw = 1.0)

plt.title("Ven diagram of Exact Matches for 3 Runs")
plt.show()

gl.savefig(folder_images + "ven3.png",  dpi = 100, sizeInches = [8, 5], close = False, bbox_inches = "tight") 
   
intersection = np.sum(np.bitwise_and(np.bitwise_and(DataSet_statistics_lists[0]["em"], DataSet_statistics_lists[1]["em"]), DataSet_statistics_lists[2]["em"]))
EM_ensemble = np.sum(np.bitwise_or(np.bitwise_or(DataSet_statistics_lists[0]["em"], DataSet_statistics_lists[1]["em"]),DataSet_statistics_lists[2]["em"]) )
EM_ensemble_percentage = EM_ensemble/len(DataSet_statistics_lists[0]["em"])

print ("Union 3 Runs: ", EM_ensemble, EM_ensemble_percentage)