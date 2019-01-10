"""
This file loads:
    - It instantiates the ELMO + BiDAF architecture.
    - It loads from disk the parameters
    - It loads the dataset
    - It loads the training information (pickle)

Once it has all this information it plots:
    - Dataset Analysis (lengths, questions...)
    - Models error in function of length, type of word...
    - Single analysis of the Attention Matrix and Results of a given (question, passage)

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
from bidaf_model import BidirectionalAttentionFlow_1
from squad1_reader import Squad1Reader, load_SQUAD1_dataset
from bidaf_utils import print_conf_params, fill_evaluation_data,question_types
import pyTorch_utils as pytut
import pickle_lib as pkl
import utilities_lib as ul
from graph_lib import gl
import matplotlib.pyplot as plt
import squad_plotting as spl
import pandas as pd
import scipy.stats as st
from other_func_bidaf import *
plt.close("all")
"""
#################### EXECUTION FLAGS ####################
"""
Experiments_load_dataset_from_disk = 1
Experiments_instantiate_model = 1

Experiments_generate_results_data = 0 # Get the dataset properties
Evaluate_Model_Results = 1           # Run the model through the dataset

# Option
train_validation_dataset =  False # train = True, Validation = False
bayesian_ensemble = False # Either Bayesian ensemble
## Plotting options
database_lengths_plot = 0
database_question_type_plot = 0
model_performances_plot = 0

analyze_example_query = 0
trim_VB_model = True
## Folder
folder_images = "../pics/Thesis/Eval_model_dataset/"
images_prefix = "Baseline_" # We add this prefix to the image names to avoid collision
ul.create_folder_if_needed(folder_images)
"""
#################### MODEL TO LOAD ####################
Information to load the experiments
"""
#source_path = "../Baseline_model/results/bidaf/"
#model_id = "2018-11-28 01:27:02.281959" 
#source_path = "../data_VB2/CV_bayesian_2/results/bidaf/"
#model_id = "2018-12-06 19:58:11.727693" 

source_path = "../data_VB_1H/results/bidaf/"
model_id = "2018-12-07 22:12:05.592914"   # 2018-12-08 08:54:34.545417

#  "2018-10-10 21:16:33.652684"
 # "2018-09-30 00:29:50.035864" #  "2018-09-26 20:41:06.955179"  # "2018-09-27 08:36:18.902846"# "2018-09-26 20:41:06.955179" 
epoch_i = -1   # Epoch to load

pickle_results_path,model_file_path = pytut.get_models_paths(model_id, 
                                epoch_i = -1,source_path = source_path)

"""
############# Add the positibility of filtering by desired properties !!
"""
pickle_results_path = None
model_file_path = None

source_path = "../all_Bayesian/" 
pickle_results_path, models_path = pytut.get_all_pickles_training(source_path, include_models = True)
Nmodels = len(pickle_results_path)
cf_a_list = []
training_logger_list = []
for i in range(Nmodels):
    [cf_a,training_logger] = pkl.load_pickle(pickle_results_path[i])
    ## Set data and device parameters
    cf_a_list.append(cf_a)
    training_logger_list.append(training_logger)
        
        
List_analyses = ["Layers DOr","LSTMs hidden size", r"$\zeta$","sigma2",
                 "Run"]
List_f_analyses = [return_layers_dropout, return_hidden_size,return_etaKL,return_sigma2,
                   return_initialization_index]

Standard_values_alyses = [0., 100, 0.001, 0.5,"whatever"]

N_analyses = len(List_analyses)
for i in range (Nmodels):
    training_logger = training_logger_list[i]
    cf_a = cf_a_list[i]
    
    condition_analysis = True
    # Substract one because the last one is the iterations one.
    for an_j in range(N_analyses -1):
        condition_analysis &= List_f_analyses[an_j](cf_a) == Standard_values_alyses[an_j]
        
    if (condition_analysis):
           pickle_results_path = pickle_results_path[i]
           model_file_path = models_path[i]


"""
##################################################################
LOAD THE CONFIGURATION FILE
##################################################################
"""
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)
cf_a,training_logger = pkl.load_pickle(pickle_results_path)
cf_a.dtype = dtype  # Variable types
cf_a.device = device

## Modify these parameters so that I am not fucked in memory in my litle servergb
cf_a.datareader_lazy = True # Force lazyness for RAM optimization
cf_a.batch_size_train = 30
cf_a.batch_size_validation = 30 
cf_a.force_free_batch_memory = False
max_instances_in_memory = 100
print_conf_params(cf_a)

print ("Expected EM: ", 100*np.array(training_logger["validation"]["em"]))
print ("Expected F1: ", 100*np.array(training_logger["validation"]["f1"]))
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
if (Experiments_instantiate_model):
    print ("Initializing Model architecture")
    model = BidirectionalAttentionFlow_1(vocab, cf_a)
    print("Loading previous model")
    model.load_state_dict(torch.load(model_file_path))
    model.to(device = device, dtype = dtype)

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

if (trim_VB_model):
    trim_mu_sigma = [0,0.5,1,1.5,2,2.5,3.0,3.5,4.0,4.5,5.0]
#    trim_mu_sigma = [0.5,5.0]
    DataSet_statistics_list = []
    weights_statistics_list = [] # [trim_range][layer][size_w, removed_w, size_b, removed_b]
    for i in range(len(trim_mu_sigma)):
        weights_statistics = model.trim_model(trim_mu_sigma[i])
        weights_statistics_list.append(weights_statistics)
        DataSet_statistics = fill_evaluation_data(model,device, dataset_iterable,num_batches, Evaluate_Model_Results, bayesian_ensemble = bayesian_ensemble)
        DataSet_statistics_list.append(DataSet_statistics)
    DataSet_statistics = DataSet_statistics_list[0]

EM_list = []
F1_list = []
for i in range(len(DataSet_statistics_list)):
    EM_list.append(100*np.mean(DataSet_statistics_list[i]["em"]))
    F1_list.append(100*np.mean(DataSet_statistics_list[i]["f1"]))
gl.init_figure();
ax1 = gl.subplot2grid((3,1), (0,0), rowspan=1, colspan=1)
ax2 = gl.subplot2grid((3,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
ax3 = gl.subplot2grid((3,1), (2,0), rowspan=1, colspan=1,sharex = ax1)

## Accuracies !!
gl.plot(trim_mu_sigma, EM_list, ax = ax1, legend = ["EM"],
        labels = ["Trimming of the weights and biases by $|\mu_w|/\sigma_w$","","Accuracy"])
gl.plot(trim_mu_sigma, F1_list, ax = ax1, legend = ["F1"])

## Systems !
list_all_labels = ["$W_{E}$","$W_{H}$","$W_{S}$","$W_{(p^1)}$","$W_{(p^2)}$"]
list_final_pcts_w = []
list_final_pcts_b = []
for i in range(len(list_all_labels)):
    list_final_pcts_w.append([])
    list_final_pcts_b.append([])
    for j in range(len(DataSet_statistics_list)):
        list_final_pcts_w[i].append(100*(1 - float(weights_statistics_list[j][1][i])/weights_statistics_list[j][0][i]))
        list_final_pcts_b[i].append(100*(1-float(weights_statistics_list[j][3][i])/weights_statistics_list[j][2][i]))

gl.colorIndex = 0
## Weights !!
for i in range(len(list_all_labels)):
    gl.plot(trim_mu_sigma, list_final_pcts_w, ax = ax2, legend = [list_all_labels[i] + " (" + str(weights_statistics_list[0][0][i]) + ")"],
            labels = ["","","Weights"])
gl.colorIndex = 0
## Bias !!
for i in range(len(list_all_labels)):
    gl.plot(trim_mu_sigma, list_final_pcts_b, ax = ax3, legend = [list_all_labels[i] + " (" + str(weights_statistics_list[0][2][i]) + ")"],
            labels = ["",r"$|\mu_w|/\sigma_w$","Biases"])

# Set final properties and save figure
gl.set_fontSizes(ax = [ax1,ax2,ax3], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 10, xticks = 15, yticks = 15)
gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
gl.savefig(folder_images + images_prefix  +"Trimming_accuracies.png", 
        dpi = 100, sizeInches = [18, 6], close = False, bbox_inches = "tight")    
    
#elif (Experiments_generate_results_data):
#    DataSet_statistics = fill_evaluation_data(model,device, dataset_iterable,num_batches, Evaluate_Model_Results, bayesian_ensemble = bayesian_ensemble)
#    EM = 100*np.mean(DataSet_statistics["em"])
#    F1 = 100*np.mean(DataSet_statistics["f1"])
        
        
#metrics, data_loss = general_validation_runner(model)
#print ("Loss validation:",data_loss)
#print ("metrics: ", metrics)



"""
##################################################################################
################# ANALYZE THE SYSTEM INTERNALS FOR A QUERY EXAMPLE ###################
####################################################################################
"""

if (analyze_example_query):
    """
    ############  Propagate an instance text #############
    """
    
    sentence = "Ingrid really loves bacon because it is so crispy and it only takes 2 minutes to make a lot of bacon on Fridays and 10 minutes on Mondays in the microwave oven and then dinner is ready."
    question = "Why does she like bacon?"
    question = "How much does that girl takes to cook the food?"
    question = "How much does that girl takes to cook on Fridays?"
    question = "Where is the bacon?"
#    question = "Who makes bacon?"
    instance = squad_reader.text_to_instance(question, sentence, 
                                               char_spans=[(6, 10)])

#    instance = squad_reader.text_to_instance("Why the Spanish team won the Word Cup?", 
#                                               "After so many year trying to win the World Cup, failing in 1982, 1986 and so on, finally \
#                                               a goal from Iniesta in 2010 in the minute 196 gave the victory to the Spanish team.", 
#                                               char_spans=[(6, 10)])
    
    print ("Keys instance: ", instance.fields.keys())
    
    # Batch intances and convert to index using the vocabulary.
    instances = [instance]
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    
    # Create the index tensor from the vocabulary.
    cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict(cuda_device=cuda_device)
    
    # Propagate the sample and obtain the loss (since we passed labels)
    
    outputs = model(**model_input, get_attentions = True)
    
    print (outputs["best_span_str"])
    question_tokens = model_input["metadata"][0]["question_tokens"]
    passage_tokens = model_input["metadata"][0]["passage_tokens"]
    C2Q_attention = outputs["C2Q_attention"][0,:,:].detach().cpu().numpy()
    Q2C_attention = outputs["Q2C_attention"].detach().cpu().numpy()
    simmilarity_matrix = outputs["simmilarity"][0,:,:].detach().cpu().numpy()
    span_start_probs = outputs["span_start_probs"].detach().cpu().numpy()
    span_end_probs = outputs["span_end_probs"].detach().cpu().numpy()
    """
    The attention matrix is normalized by passage word, 
    every passage word has a probability distribution over the query words.
    """
    image_path = folder_images + images_prefix + "Attention_model.png"
    
    spl.visualize_attention_matrix(question_tokens, passage_tokens, C2Q_attention.T,
                               image_path)
    image_path = folder_images + images_prefix + "probs_end.png"
    spl.visualize_solution_matrix(passage_tokens, np.concatenate((span_start_probs,span_end_probs)),outputs["best_span_str"][0],
                               image_path)

    image_path = folder_images + images_prefix + "similarity.png"
    spl.visualize_similarity_matrix(question_tokens, passage_tokens, simmilarity_matrix.T,
                               image_path)
    
    image_path = folder_images + images_prefix + "Attention_model_C2Q.png"
    spl.visualize_attention_matrix(["Q2C_attention"], passage_tokens, Q2C_attention,
                               image_path)


"""
########################################################################
################# PLOT THE DATABASE ONLY INFORMATION ###################
########################################################################
"""

if (database_lengths_plot):
    gl.init_figure();
    ax1 = gl.subplot2grid((1,3), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((1,3), (0,1), rowspan=1, colspan=1)
    ax3 = gl.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)
        
    ############## Context length ###############
    spl.distplot_1D_seaborn(DataSet_statistics["passage_length"], ax1,
                        labels = ['Distribution of Context Length',"",''])
    
    ############## Question length ###############
    spl.distplot_1D_seaborn(DataSet_statistics["question_length"], ax2,
                        labels = ['Distribution of Question Length',"",''])
    
    ############## Answer Span length ###############
    spl.distplot_1D_seaborn(DataSet_statistics["span_length"], ax3,
                        labels = ['Distribution of Answer Span Length',"",''])
    
    # Set final properties and save figure
    gl.set_fontSizes(ax = [ax1,ax2,ax3], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 10, xticks = 15, yticks = 15)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    gl.savefig(folder_images + images_prefix  +"Lengths_dataset.png", 
            dpi = 100, sizeInches = [18, 6], close = False, bbox_inches = "tight")
    ####################################################################################
    ### Relative index promotion
    gl.init_figure();
    ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
    ############## Answer Start Span Position ###############
    spl.distplot_1D_seaborn(DataSet_statistics["span_start"],ax1 ,
                        labels = ['Distribution of Span Start Position',"",''])
    
    ############## Answer Start Span Position Index ###############
    relative_span_start = np.array(DataSet_statistics["span_start"])/np.array(DataSet_statistics["passage_length"])
    spl.distplot_1D_seaborn(relative_span_start, ax2,
                        labels = ['Distribution of Span Start Relative Position',"",''])
    
    gl.set_fontSizes(ax = [ax1,ax2,ax3], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 10, xticks = 15, yticks = 15)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    gl.savefig(folder_images + images_prefix + "StartPoint_dataset.png", 
            dpi = 100, sizeInches = [18, 6], close = False, bbox_inches = "tight")
    
if (database_question_type_plot):
    ############## By type of question  ##############
    
    question_types_ammount = []
    question_types_list = np.array(DataSet_statistics["question_type"]).flatten()
    for i in range(len(question_types)):
        question_types_ammount.append(np.where(question_types_list == i)[0].size)

    ## Normalize and order
    question_types_ammount, question_types_ordered = zip(*sorted(zip(question_types_ammount, question_types),reverse = True))
    question_types_ammount = np.array(question_types_ammount)
    question_types_ammount = question_types_ammount/np.sum(question_types_ammount)
    
    ## Do the actual plotting
    gl.init_figure()
    ax1 = gl.bar(question_types_ordered,question_types_ammount, align = "center", 
                 labels = ["Question Types Proportions SQUAD 1.1","","Proportion"])
    gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 10, xticks = 15, yticks = 15)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    gl.savefig(folder_images + images_prefix + "Question_Types_dataset.png", 
            dpi = 100, sizeInches = [18, 6], close = False, bbox_inches = "tight")

"""
########################################################################
####### PLOT PERFORMANCE OF THE MODEL IN TERMS OF OTHER PARAMS  ########
########################################################################
"""

def get_mean_variance_in_buckets(x,y, min_value = None,  max_value = None, 
                                 nbins = 10, step_size = 2, type_x = "int"):

    if(type(min_value) == type(None)):
        min_value = np.min(x)
    if(type(max_value) == type(None)):
        max_value = np.max(x)  
#    print ("min: ",min_value, ", max: ",max_value)
    
    if (type_x == "int"):
        x_values = range(min_value,max_value, step_size)
    else:
        x_values = np.linspace(min_value,max_value, nbins, endpoint = False)
    
        
    y_means = []
    y_stds = []
    x_labels = []
    y_s = []
    
    y = y.flatten()
    
    for  i in range(len(x_values)):
        if (type_x == "int"):
            if (i != len(x_values)-1): # Last bin
                y_range_values = y[np.bitwise_and(x >= x_values[i],x < x_values[i+1])]
                if (step_size > 1):
                    x_range_values = "%i - %i"%(np.ceil(x_values[i]),np.floor(x_values[i+1]))
                else:
                    x_range_values = "%i"%(np.ceil(x_values[i]))
        
            else:
                y_range_values = y[x >= x_values[i]]
                if (step_size > 1):
                    x_range_values = "%i - %i"%(np.ceil(x_values[i]), max_value)
                else:
                    x_range_values = "%i"%(np.ceil(x_values[i]))
        else:
            if (i != len(x_values)-1): # Last bin
                y_range_values = y[np.bitwise_and(x >= x_values[i],x < x_values[i+1])]
                x_range_values = "[%.2f - %.2f)"%(x_values[i],x_values[i+1])
        
            else:
                y_range_values = y[x >= x_values[i]]
                x_range_values = "[%.2f - %.2f)"%(x_values[i], max_value)
        
        y_s.append(y_range_values)
        y_means.append(np.mean(y_range_values))
        y_stds.append(np.std(y_range_values)) 
        x_labels.append(x_range_values)
    return x_labels,y_s,y_means,y_stds


if (model_performances_plot):
    
    ########################################################
    ############## By Question type  ##############
    
    question_types_f1 = []
    question_types_em = []
    question_types_loss = []
    question_types_list = np.array(DataSet_statistics["question_type"]).flatten()
    for i in range(len(question_types)):
        samples_indx = np.where(question_types_list == i)[0]
        # start_span_loss end_span_loss em f1
        question_types_f1.append(np.mean(np.array(DataSet_statistics["f1"])[samples_indx]))
        question_types_em.append(np.mean(np.array(DataSet_statistics["em"])[samples_indx]))
        question_types_loss.append(np.mean(np.array(DataSet_statistics["start_span_loss"])[samples_indx] + np.array(DataSet_statistics["end_span_loss"])[samples_indx]))  
   
    ## Normalize and order. We order according to F1. We also have to reorder the EM
    question_types_f1_ordered, question_types_ordered = zip(*sorted(zip(question_types_f1, question_types),reverse = True))
    question_types_f1_ordered, question_types_em_ordered = zip(*sorted(zip(question_types_f1, question_types_em),reverse = True))
    question_types_f1_ordered, question_types_loss_ordered = zip(*sorted(zip(question_types_f1, question_types_loss),reverse = True))
 
    question_types_f1_ordered = np.array(question_types_f1_ordered)
    question_types_em_ordered = np.array(question_types_em_ordered)
    question_types_loss_ordered =  np.array(question_types_loss_ordered)
    
    Ntypes_questions = len(question_types_ordered)
    
    ## Do the actual plotting
    gl.init_figure()
    ax1 = gl.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
    plt.bar(np.arange(Ntypes_questions)+0.2,question_types_f1_ordered,width=0.2,color='c',align='center', label = "F1")
    plt.bar(np.arange(Ntypes_questions)+0.4,question_types_em_ordered,width=0.2,color='r',align='center', label = "EM")
#    plt.bar(np.arange(Ntypes_questions)+0.6,question_types_loss_ordered,width=0.2,color='g',align='center', label = "loss")
    
    plt.xticks(np.arange(Ntypes_questions)+0.3,question_types_ordered)
    plt.title('Accuracy in terms of question types')
    plt.ylabel('Accuracy')
    plt.grid()

    gl.set_fontSizes(ax = [ax1], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 15, yticks = 15)
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
    gl.savefig(folder_images + images_prefix +"Question_Types_performance.png", 
            dpi = 100, sizeInches = [18, 4], close = False, bbox_inches = "tight")
    
    ########################################################
    ############## By Absolute Span start  ##############
    
    type_analysis = ["span_start", "relative_span_start", "passage_length","question_length", "span_length"]
    
    for type_anal in type_analysis:
        if (type_anal == "span_start"):
            x_values = relative_span_start = np.array(DataSet_statistics["span_start"])
            labels_1 = ["Loss in terms of the span start ","Span start absolute position","Loss"]
            labels_2 = ["Boxplot of the Loss and mean accuracy measures by span start absolute position"]
            image_name_type_x = "Performance_to_absolute_span_start"
            max_value_x = 60; step_size = 1; type_x = "int"; nbins = None; min_value_x = None
        elif (type_anal == "relative_span_start"):
            x_values = relative_span_start = np.array(DataSet_statistics["span_start"])/np.array(DataSet_statistics["passage_length"])
            labels_1 = ["Loss in terms of the relative span start ","Span start relative position","Loss"]
            labels_2 = ["Boxplot of the Loss and mean accuracy measures by span start relative position"]
            image_name_type_x = "Performance_to_relative_span_start"
            max_value_x = None; step_size = 1; type_x = "float"; nbins = 10; min_value_x = None
        elif (type_anal == "passage_length"):
            x_values = relative_span_start = np.array(DataSet_statistics["passage_length"])
            labels_1 = ["Loss in terms of the passage length ","Passage length","Loss"]
            labels_2 = ["Boxplot of the Loss and mean accuracy measures by passage length"]
            image_name_type_x = "Performance_to_relative_passage_length"
            max_value_x = 300; step_size = 5; type_x = "int"; nbins = None; min_value_x = None
            
        elif (type_anal == "span_length"):
            x_values = relative_span_start = np.array(DataSet_statistics["span_length"])
            labels_1 = ["Loss in terms of the span length ","Span length","Loss"]
            labels_2 = ["Boxplot of the Loss and mean accuracy measures by span length"]
            image_name_type_x = "Performance_to_relative_span_length"
            max_value_x = None; step_size = 1; type_x = "int"; nbins = None; min_value_x = None
        elif (type_anal == "question_length"):
            x_values = relative_span_start = np.array(DataSet_statistics["question_length"])
            labels_1 = ["Loss in terms of the question length ","Question length","Loss"]
            labels_2 = ["Boxplot of the Loss and mean accuracy measures by question length"]
            image_name_type_x = "Performance_to_relative_question_length"
            max_value_x = None; step_size = 1; type_x = "int"; nbins = None; min_value_x = None
            
        gl.init_figure()
        ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
        ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1, sharex = ax1)
        
        y_values = np.array(DataSet_statistics["start_span_loss"]) +  np.array(DataSet_statistics["end_span_loss"])
        
        y_EM_values = np.array(DataSet_statistics["em"])
        y_F1_values = np.array(DataSet_statistics["f1"])
        
        ax1 = gl.scatter(x_values,y_values, ax = ax1,
                         labels = labels_1, alpha = 0.1)
        spl.distplot_1D_seaborn(x_values,ax2 ,
                        labels = ['Distribution of '+labels_1[0],labels_1[1],'pdf('+labels_1[1] + ")"])
        gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 15, ylabel = 18, 
                          legend = 10, xticks = 15, yticks = 15)
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
        gl.savefig(folder_images +images_prefix+image_name_type_x + ".png", 
                dpi = 100, sizeInches = [18, 4], close = False, bbox_inches = "tight")
        """
       SECOND CHART !!!
        """
    #    y_values = span_start_loss = np.array(DataSet_statistics["em"])  
        x_labels,y_s,y_means,y_stds = get_mean_variance_in_buckets(x_values,y_values, min_value = min_value_x, max_value = max_value_x,
                                                                   nbins = nbins, step_size = step_size,type_x = type_x )
        
        x_labels,y_s_em,y_means_em,y_stds_em = get_mean_variance_in_buckets(x_values,y_EM_values, min_value = min_value_x, max_value = max_value_x,
                                                                   nbins = nbins, step_size = step_size,type_x = type_x )
        x_labels,y_s_f1,y_means_f1,y_stds_f1 = get_mean_variance_in_buckets(x_values,y_F1_values, min_value = min_value_x, max_value = max_value_x,
                                                                   nbins = nbins, step_size = step_size,type_x = type_x )
        
        ########################################################
        ############## Answer Span length ###############
        gl.init_figure()
        ax1 = gl.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
        ax2 = gl.subplot2grid((2,1), (1,0), rowspan=1, colspan=1, sharex = ax1)
#        ax1 = gl.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
#        ax2 = gl.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
        
        
        ## Ca
        bp = ax1.boxplot(y_s,patch_artist=True)
#        ax1.set_xticklabels(None)
        ax1.set_title(labels_2[0])
        ax1.set_ylabel(labels_1[2])
        
        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set( color='#7570b3', linewidth=2)
            # change fill color
            box.set( facecolor = '#1b9e77' )
        
        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)
        
        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)
        
        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)
        
        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)
              
        gl.plot(x_labels, y_means_em, ax = ax2, legend = ["EM"],
                labels = ["",labels_1[1],"Accuracy"])
        gl.plot(x_labels, y_means_f1, ax = ax2, legend = ["F1"])
        
#        ax2= ax1
        if (0):
            ## 2D distribution !!
            xmin = min(x_values); xmax = max(x_values)
            ymin = min(y_values); ymax = np.mean(y_values) + 2*np.std(y_values)
            # Peform the kernel density estimate
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x_values, y_values])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            
            ax = ax2
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            # Contourf plot
            cfset = ax.contourf(xx, yy, f, cmap='Blues')
            ## Or kernel density estimate plot instead of the contourf plot
            #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
            # Contour plot
            cset = ax.contour(xx, yy, f, colors='k')
            # Label plot
            ax.clabel(cset, inline=1, fontsize=10)
            ax.set_xlabel('Y1')
            ax.set_ylabel('Y0')
            
        gl.set_zoom(ax = ax1, ylim = [-2,25])

        gl.set_fontSizes(ax = [ax1,ax2], title = 18, xlabel = 15, ylabel = 18, 
                             legend = 18, xticks = 10, yticks = 15)
        
        ax2.set_xticklabels(x_labels, rotation = 45, ha="right")
      
        plt.setp(ax1.get_xticklabels(), visible=False)
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.00, hspace=0.10)
            
        gl.savefig(folder_images +images_prefix+image_name_type_x+ "_acc.png", 
                 dpi = 100, sizeInches = [18, 6], close = False, bbox_inches = "tight")
        
