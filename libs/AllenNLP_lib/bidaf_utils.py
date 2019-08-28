import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common import squad_eval

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

"""
OWN LIBRARY
"""
from GeneralVBModelRNN import GeneralVBModelRNN
import pyTorch_utils as pytut
# @Model.register("bidaf1")
"""
EMBEDDING LIBRARIES
"""
from allennlp.modules.elmo import Elmo

import Variational_inferences_lib as Vil
import numpy as np
from pydoc import locate  # To convert types
import ast
import gc

question_types = ["What", "When","How many","How much", "How", "In",
                 "Whose","Who", "Where","Which","Why", "Be/Do/etc"]
class config_bidaf():
    
    """
    #########################################################
    ############# DATASET PARAMETERS ###################
    """
    train_squad1_file = "../data/squad/train-v1.1.json"
    validation_squad1_file = "../data/squad/dev-v1.1.json"
    datareader_lazy = True # We cannot load all of these data in RAM so we need this

    results_root_folder = "../" # /scratch 
    ## Set automatically during training to know directly the path of the results
    mode_file_path = None # /scratch 
    """
    #########################################################
    ############# ARCHITECTURE PARAMETERS ###################
    """
    # Embedding parameters
    tokenizer_indexer_type = "elmo"
    use_ELMO = True
    ELMO_num_layers = 2  # The last one is added
    ELMO_droput = 0.2
    
    ## Projection to d = 200
    Add_Linear_projection_ELMO = True
    
    # Highway parameters
    num_highway_layers = 1
    
    ## Phrase layer parameters
    phrase_layer_dropout = 0.2
    phrase_layer_num_layers = 1
    phrase_layer_hidden_size = 100
    
    # Modelling Passage parameters
    modeling_passage_dropout = 0.2
    modeling_passage_num_layers = 2
    modeling_passage_hidden_size = 100
    
    # Span end encoding parameters
    span_end_encoder_dropout = 0.2
    modeling_span_end_num_layers = 1
    modeling_span_end_hidden_size = 100
    
    # Masking parameters
    mask_lstms = True
    
    # spans output parameters
    spans_output_dropout  = 0.2

    """
    #####################################################
    ############# TRAINING PARAMETERS ###################
    """
    # Initializer,optimizer and regularizer
    regularizer = None
    
    # Optimizer parameters
    optimizer_type = "Adam"
    optimizer_params = {"lr":1e-3, "betas":(0.9, 0.9)}
    
    # Training algorithm configuration
    num_epochs = 10 # Maximum number of epochs
    batch_size_train = 30
    batch_size_validation = 30

    ### Parameters to use if the loading is lazy ###
    # Since the lazy mode does not know how many instances, this will do
    instances_per_epoch_train = 87599 #  87599 1000
    instances_per_epoch_validation = 10570 # 10570 1000
    # We can preload a subset of them so that they are shuffled and optmized lengths
    # It depends on how much RAM you have.
    max_instances_in_memory = 10000
    ###  Memory management ###
    # Someties the garbage collector is not good at freeing memmory on time,
    # If we are constrained in memory, we might want to force the removal.
    # NOTE: In the servers with big RAM, calling gc.collect() will take too much time.
    
    force_free_batch_memory = True # Will avoid unnecesary out of memory
    force_call_garbage_collector = False

    """
    #####################################################
    ############# BAYESIAN PARAMETERS ###################
    """
    
    VB_Linear_projection_ELMO = False
    VB_highway_layers = False
    VB_similarity_function = False
    VB_span_start_predictor_linear = False
    VB_span_end_predictor_linear = False 

    VB_Linear_projection_ELMO_prior  = \
    {"pi":0.5, "log_sigma1":np.log(0.5), "log_sigma2":np.log(0.5)}
    VB_highway_layers_prior  = \
    {"pi":0.5, "log_sigma1":np.log(0.5), "log_sigma2":np.log(0.5)}
    VB_similarity_function_prior  = \
    {"pi":0.5, "log_sigma1":np.log(0.5), "log_sigma2":np.log(0.5)}
    VB_span_start_predictor_linear_prior  = \
    {"pi":0.5, "log_sigma1":np.log(0.5), "log_sigma2":np.log(0.5)}
    VB_span_end_predictor_linear_prior  = \
    {"pi":0.5, "log_sigma1":np.log(0.5), "log_sigma2":np.log(0.5)}


    # Weight applied to tbe KL 
    eta_KL = 0
    Nsamples_train = instances_per_epoch_train
    
    """
    #####################################################
    ############# SAVING TRAINING INFO OPTIONS ##########
    We save the weights of the entire model (maybe exclude ELMO in the future)
    every epoch. We can decide weather to keep them or just the last one.
    We save the also the cumulative training accuracies and the validation accuracy.
    We can also save the the losses and cummulative accuracies every a few epochs
    
    Also save the variational weights for later analysis.
    """
    # Every this number of batches we log t
    logging_lossesMetrics_batch_frequency = 5 
    save_weights_at_each_epoch = True
    save_only_last_weights = True # Only save the last set of parameters
    logging_VB_weights_batch_frequency = 10
    
    """
    FUTURE PARAMETERS !!!!!
    At some point you will probably want to add more parameters
    to this configuration. Due to the pickle saving, this could conflict
    when loading previous versions that do not have those parameters, therefore
    it is advised to put new parameters in the new dictionarty
    """
    
    future_extensions_dict = None
    command_line_args = None
    
def init_DataSet_statistics():
      DataSet_statistics = dict()
      # Only dataset related
      DataSet_statistics["question_length"] = []
      DataSet_statistics["passage_length"] = []
      DataSet_statistics["span_length"] = []
      DataSet_statistics["span_start"] = []
      DataSet_statistics["question_type"] = []
      DataSet_statistics["answer"] = []

      # Model related
      DataSet_statistics["start_span_loss"] = []
      DataSet_statistics["end_span_loss"] = []
      DataSet_statistics["em"] = []
      DataSet_statistics["f1"] = []
      DataSet_statistics["estimated_answer"] = []
      return DataSet_statistics

def get_quetion_type(string_sentence):
    # We assume the question starts with the question type
    for i in range(len(question_types)):
        if (question_types[i] in string_sentence):
#            print (string_sentence)
            return i
        # If not found, we assume it is a question starting with Do Does Is Are
    return i

def fill_DataSet_statistics_from_metadata(tensor_dict, DataSet_statistics):
    """
      This function fills the DataSet_statistics from the metadata of a single sample
    """
    metadata_list = tensor_dict["metadata"] # batch length
    span_start = tensor_dict["span_start"]
    span_end = tensor_dict["span_end"]
    
    for i in range(len(metadata_list)):
        m = metadata_list[i]
        DataSet_statistics["question_length"].append(len(m["question_tokens"]))
        DataSet_statistics["passage_length"].append(len(m["passage_tokens"]))
        DataSet_statistics["span_length"].append(span_end[i]- span_start[i] +1)
        DataSet_statistics["span_start"].append(span_start[i])
        DataSet_statistics["question_type"].append(get_quetion_type(" ".join(m["question_tokens"])))
        DataSet_statistics["answer"].append(m.get('answer_texts', []))
        
def fill_DataSet_statistics_from_model_results(output_batch, DataSet_statistics):
    """
      This function fills the DataSet_statistics from the metadata of a single sample
    """
    
    DataSet_statistics["start_span_loss"].extend(output_batch["span_start_sample_loss"])
    DataSet_statistics["end_span_loss"].extend(output_batch["span_end_sample_loss"])
    DataSet_statistics["em"].extend(output_batch["em_samples"])
    DataSet_statistics["f1"].extend(output_batch["f1_samples"])
    DataSet_statistics["estimated_answer"].extend(output_batch["best_span_str"])

def fill_evaluation_data(model,device, dataset_iterable,num_batches, Evaluate_Model_Results,bayesian_ensemble = True):
    
    """
    Extended:
        If model is a list, then we compute different DataSet_statistics for each model.
        For each sample, we move the models into GPU one at a time.
    """
    if (type(model) != type([])): # List of model
        model = [model]
    DataSet_statistics = []
    for i in range (len(model)):
        DataSet_statistics.append(init_DataSet_statistics())
        cf_a = model[i].cf_a
        model[i].set_posterior_mean(True)
        model[i].eval()

    with torch.no_grad():
        """
        Extended to be able to analyze several models. 
        If only one model given we put it into a list.
        """
        # Compute the validation accuracy by using all the Validation dataset but in batches.
        for j in range(num_batches):
            print ("Batch %i/%i"%(j,num_batches))
            tensor_dict = next(dataset_iterable)
          
            ############# fill data for the dataset ###########

            for i in range (len(model)):
                fill_DataSet_statistics_from_metadata(tensor_dict, DataSet_statistics[i])
      
            if (Evaluate_Model_Results):
                tensor_dict = pytut.move_to_device(tensor_dict, device) ## Move the tensor to cuda
            
                for i in range (len(model)):
                    if(bayesian_ensemble):
                        output_batch = model[i].forward_ensemble(**tensor_dict,get_sample_level_information = True)
                    else:
                        output_batch = model[i].forward(**tensor_dict,get_sample_level_information = True)
      
                    fill_DataSet_statistics_from_model_results(output_batch, DataSet_statistics[i])
                    del output_batch
                    torch.cuda.empty_cache()
    
            ## Memmory management !
            if (cf_a.force_free_batch_memory):
                del tensor_dict["question"]; del tensor_dict["passage"]
                del tensor_dict
                torch.cuda.empty_cache()
            if (cf_a.force_call_garbage_collector):
                gc.collect()
                
    if (type(model) == type([])): # List of model
        for i in range (len(model)):
            model[i].train()
            model[i].set_posterior_mean(False)
    else:
        model.train()
        model.set_posterior_mean(False)

    if len(DataSet_statistics) == 1:
        DataSet_statistics = DataSet_statistics[0]
    return DataSet_statistics


    
def print_conf_params (config_architecture):
    config_params = dir(config_architecture)
    print ("------ Architecture Parameters -------")
    for param in config_params:
        if (param[0] != "_"):
            value =  getattr(config_architecture, param)
            print (param, " = ",value)
            
def parse_args_architecture(args, config_architecture):
    """
    Function that gets the list of args from the console. 
    It will
    """
    # Save the command line parameters for later knowing what exactly we changed.
    config_architecture.command_line_args = args
    config_params = dir(config_architecture)
    for arg in args:
        arg_splitted = arg.split("=")
        if (len(arg_splitted)):
            arg_name = arg_splitted[0]
            arg_value_str = arg_splitted[1]
            
            if (arg_name in config_params):
                print("Setting parameter %s to %s by console"%(arg_name, arg_value_str))
                prev_value = getattr(config_architecture, arg_name)

                new_value = string2value(arg_value_str, prev_value)
                
                setattr(config_architecture, arg_name, new_value)
    return config_architecture
def string2value(arg_value_str, example_value = None):
    """
    TODO: Complete for possibility of anidated lists.
    Convert the string parameters to python structures.
    There is the problem that the " from the dictionary specifications is removed
    so we reset them ourselves
    """
    if (is_dict(arg_value_str)):   # Dictionary
        """
        Since bloody command line removes anidated strings, I will first obtain every parameter and
        their value looking for ":" and "," and apply recurrenty until this works.
         Format: {lr:1e-3,betas:(0.9,0.9)}
        """
        s = arg_value_str[1:] # Remove {
        new_value = dict()
        while (1): ## While more parameters
            param_name = s[:s.find(':')]
            end = s.find(',')
            if (end == -1):
                end = s.find('}')
    
            ## Check if the value is gonna be a list, typle or dict
            if (s[s.find(':')+1] == "{"):
                end = s.find('}')+1
            if (s[s.find(':')+1] == "("):
                end = s.find(')')+1
            if (s[s.find(':')+1] == "["):
                end = s.find(']')+1
                
            param_value = string2value(s[s.find(':')+1:end])
            # Add to the dictionary
            new_value[param_name] = param_value
            # Remove parameters
            if (s.find(',') == -1):
                break
            s = s[end+1:]
#            print ("Remaining dict: ", s)
            if(len(s) == 0):
                break
    elif(is_list(arg_value_str)):
        s = arg_value_str[1:] # Remove []
        new_value = list()
        
        while (1): ## While more parameters
            end = s.find(',')
            if (end == -1):
                end = s.find(']')
            ## Check if the value is gonna be a list, typle or dict
            if (s[s.find(':')+1] == "{"):
                end = s.find('}')+1
            if (s[s.find(':')+1] == "("):
                end = s.find(')')+1
            if (s[s.find(':')+1] == "["):
                end = s.find(']')+1
            param_value = string2value(s[:end])
            # Add to the dictionary
            new_value.append( param_value)
            # Remove parameters
            if (s.find(',') == -1):
                break
            s = s[end+1:]
            if(len(s) == 0):
                break
            
    elif(is_tuple(arg_value_str)):
        s = arg_value_str[1:] # Remove []
        new_value = list()
        
        while (1): ## While more parameters
            end = s.find(',')
            if (end == -1):
                end = s.find(')')
            ## Check if the value is gonna be a list, typle or dict
            if (s[s.find(':')+1] == "{"):
                end = s.find('}')+1
            if (s[s.find(':')+1] == "("):
                end = s.find(')')+1
            if (s[s.find(':')+1] == "["):
                end = s.find(']')+1
            param_value = string2value(s[:end])
            # Add to the dictionary
            new_value[param_name] = param_value
            # Remove parameters
            if (s.find(',') == -1):
                break
            s = s[end+1:]
            if(len(s) == 0):
                break
        new_value = tuple (new_value)
            
    elif(is_bool(arg_value_str)):
        new_value = arg_value_str == "True"
    elif(is_int(arg_value_str)):
        new_value =  int(arg_value_str)
    elif(is_float(arg_value_str)):
        new_value = float(arg_value_str)
    else:  # String by default
        new_value = arg_value_str

#            ### Previous implementation
#            type_example = type(example_value).__name__
#            t = locate(type_example)
#            arg_value = t(arg_value_str)

#    print ("Parsed value: ", [arg_value_str,new_value])
    return new_value


"""
CHECKING FUNCTIONS FOR PARSING
"""

def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

def is_int(input):
  try:
    num = int(input)
  except ValueError:
    return False
  return True

def is_bool(input):
  if (input == "True"):
      return True
  if (input == "False"):
      return True
  return False

def is_list(input):
  if(input[0] == "["):
      return True
  return False

def is_tuple(input):
  if(input[0] == "("):
      return True
  return False

def is_dict (input):
  if(input[0] == "{"):
      return True
  return False

def get_em_f1_metrics(best_span_string,answer_strings):
        exact_match = squad_eval.metric_max_over_ground_truths(
                squad_eval.exact_match_score,
                best_span_string,
                answer_strings)
        f1_score = squad_eval.metric_max_over_ground_truths(
                squad_eval.f1_score,
                best_span_string,
                answer_strings)
        return exact_match,f1_score
    
def init_training_logger():
    training_logger = dict() # Dictionary with all the training information
    training_logger["train"] = dict() 
    training_logger["validation"] = dict() 
    training_logger["time"] = dict() # The duration of the epoch
    
    ## Batch training dataset information
    training_logger["train"]["span_start_loss_batch"] = []
    training_logger["train"]["span_end_loss_batch"] = []
    training_logger["train"]["loss_batch"] = []

    training_logger["train"]["start_acc_batch"] = [] # Probability of getting the start span right
    training_logger["train"]["end_acc_batch"] = []  # Probability of getting the end span right
    training_logger["train"]["span_acc_batch"] = [] # Probability of getting the entire span right
    training_logger["train"]["em_batch"] = []
    training_logger["train"]["f1_batch"] = []
    
    ## Epoch train and validation information
    training_logger["train"]["start_acc"] = []
    training_logger["train"]["end_acc"] = []
    training_logger["train"]["span_acc"] = []
    training_logger["train"]["em"] = []
    training_logger["train"]["f1"] = []
    
    training_logger["validation"]["start_acc"] = []
    training_logger["validation"]["end_acc"] = []
    training_logger["validation"]["span_acc"] = []
    training_logger["validation"]["em"] = []
    training_logger["validation"]["f1"] = []
    training_logger["validation"]["data_loss"] = []
    
    training_logger["time"]["epoch"] = [] # The duration of the epoch
    training_logger["time"]["training"] = [] # The duration of the epoch
    training_logger["time"]["validation"] = [] # The duration of the epoch
    training_logger["time"]["saving_parameters"] = []
    return training_logger
    
def print_caca(model):
        print ("--------- NAMED PARAMETERS ------------")
        for f in model.named_parameters():
            print ("Name: ",f[0])
            print ("dtype: ", f[1].dtype, " device: ", f[1].device, " size: ", f[1].shape)
            print ("requires_grad: ", f[1].requires_grad)

def download_Elmo(ELMO_num_layers = 2, ELMO_droput = 0.2):
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    print ("Loading ELMO")
    text_field_embedder = Elmo(options_file, weight_file, ELMO_num_layers, dropout=ELMO_droput)
    print ("ELMO weights loaded")
    return text_field_embedder

import Cemail

class email_config():
    user = "iotubes.dk@gmail.com"
    pwd = "Iotubes1"
    recipients = ["manuwhs@gmail.com"]
    """ "maha16bf@student.cbs.dk","Christopher_wild@gmx.net",
                  "jakub.brzosko@best.krakow.pl", "s171869@student.dtu.dk","s172393@student.dtu.dk"]
    """              
    
    #"fogh.engineering@gmail.com"]
    subject = "[Thesis] Results from execution"
    body = ""

def send_email(email_config, training_logger, cf_a):
    ############### Send Email ####################
    myMail = Cemail.Cemail(email_config.user,email_config.pwd,email_config.recipients)
    myMail.create_msgRoot(subject = email_config.subject )
    #myMail.set_subject(subject)  # For some reason we can only initilize the Subject

    body_hyperparameters = "------------- SUMMARY OF SPECFIED HYPERPARAMETERS ---------- <br>"
    for i in range(len(cf_a.command_line_args)):
        body_hyperparameters += cf_a.command_line_args[i] + "<br>"
    
    body_accuracies = "-------- SUMMARY OF ACCURACIES ---------------- <br>"
    body_accuracies += "Training EM: <br>"
    for i in range(len(training_logger["train"]["em"])):
        body_accuracies += "   %.2f, "%(training_logger["train"]["em"][i])
    body_accuracies += "<br>" 
    
    body_accuracies += "Validation EM: <br>"
    for i in range(len(training_logger["validation"]["em"])):
        body_accuracies += "   %.2f, "%(training_logger["validation"]["em"][i])
    body_accuracies += "<br>"
    
    body_accuracies += "Training F1: <br>"
    for i in range(len(training_logger["train"]["f1"])):
        body_accuracies += "   %.2f, "%(training_logger["train"]["f1"][i])
    body_accuracies += "<br>" 
    
    body_accuracies += "Validation F1: <br>"
    for i in range(len(training_logger["validation"]["f1"])):
        body_accuracies += "   %.2f, "%(training_logger["validation"]["f1"][i])
    body_accuracies += "<br>"
    ########## YOU MAY HAVE TO ACTIVATE THE USED OF UNTRUSTFUL APPS IN GMAIL #####
    myMail.add_file(filedir = cf_a.pickle_results_path, filename = cf_a.pickle_results_path) # WE add the results of the train as well, why not.
    myMail.add_HTML(email_config.body + body_hyperparameters + body_accuracies)
    myMail.send_email()


def send_error_email(args):
    ############### Send Email ####################
    myMail = Cemail.Cemail(email_config.user,email_config.pwd,email_config.recipients)
    myMail.create_msgRoot(subject = email_config.subject )
    #myMail.set_subject(subject)  # For some reason we can only initilize the Subject

    body_hyperparameters = "------------- ERROR IN CLUSTER EXCUTION ---------- <br>"
    body_hyperparameters += "------------- SUMMARY OF SPECFIED HYPERPARAMETERS ---------- <br>"
    body_hyperparameters += args
    ########## YOU MAY HAVE TO ACTIVATE THE USED OF UNTRUSTFUL APPS IN GMAIL #####
 
    myMail.add_HTML(email_config.body + body_hyperparameters )
    myMail.send_email()
    
def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    max_span_log_prob = [-1e20] * batch_size
    span_start_argmax = [0] * batch_size
    best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

    span_start_logits = span_start_logits.detach().cpu().numpy()
    span_end_logits = span_end_logits.detach().cpu().numpy()

    for b in range(batch_size):  # pylint: disable=invalid-name
        for j in range(passage_length):
            val1 = span_start_logits[b, span_start_argmax[b]]
            if val1 < span_start_logits[b, j]:
                span_start_argmax[b] = j
                val1 = span_start_logits[b, j]

            val2 = span_end_logits[b, j]

            if val1 + val2 > max_span_log_prob[b]:
                best_word_span[b, 0] = span_start_argmax[b]
                best_word_span[b, 1] = j
                max_span_log_prob[b] = val1 + val2
    return best_word_span
    
def merge_span_probs(subresults: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Identifies the best prediction given the results from the submodels.
    Parameters
    ----------
    index : int
        The index within this index to ensemble
    subresults : List[Dict[str, torch.Tensor]]
    Returns
    -------
    The index of the best submodel.
    """

    # Choose the highest average confidence span.
    
    merge_type = "Addition"
    if(merge_type == "Addition"):
#        print ("Len subresults: ", len(subresults))
        span_start_probs = sum(subresult['span_start_probs'] for subresult in subresults) / len(subresults)
        span_end_probs = sum(subresult['span_end_probs'] for subresult in subresults) / len(subresults)
    else:
#        print (len(subresults))
#        print (subresults[0]['span_start_probs'])
#        print (subresults[0]['span_start_probs'].shape)
        
        spans_sumbodels = [get_best_span(subresult['span_start_probs'].log(), subresult['span_end_probs'].log()) for subresult in subresults] 
        predicted_spans = []
        batch_size = spans_sumbodels[0].shape[0]
        for i in range(len(subresults)):
            predicted_spans.append([list(spans_sumbodels[i][index].detach().cpu().numpy()) for index in range(batch_size)])
        
        answer_probs = []
        batch_size = spans_sumbodels[0].shape[0]
        for i in range(len(subresults)):
            print (subresults[i]['span_start_probs'][0,spans_sumbodels[i][0]].detach().cpu().numpy())
            answer_probs.append([float(subresults[i]['span_start_probs'][index,spans_sumbodels[i][index][0]].detach().cpu().numpy()) \
                                 *float(subresults[i]['span_start_probs'][index,spans_sumbodels[i][index][1]].detach().cpu().numpy())  for index in range(batch_size)])
#        print(answer_probs)
        
        answer_probs = np.array(answer_probs)
#        print (answer_probs.shape)
        best_model_indexes = np.argmax(answer_probs, axis= 0) # One per sample in batch
#        print ("Best Model: ", (best_model_indexes))
#        print ("All Models probabilities: ", answer_probs)
        
        ## Final subresults. We just create them as vessel
        span_start_probs = sum(subresult['span_start_probs'] for subresult in subresults) / len(subresults)
        span_end_probs = sum(subresult['span_end_probs'] for subresult in subresults) / len(subresults)
        
        for index in range(batch_size):
            span_start_probs[index,:] = subresults[best_model_indexes[index]]['span_start_probs'][index]
            span_end_probs[index,:] = subresults[best_model_indexes[index]]['span_end_probs'][index]
    
    return get_best_span(span_start_probs.log(), span_end_probs.log())

