
import numpy as np
import utilities_lib as ul
def return_hidden_size(cf_a):
    return cf_a.modeling_span_end_hidden_size

def return_sigma2(cf_a):
    return round(np.exp(cf_a.VB_span_end_predictor_linear_prior["log_sigma2"]),3)

def return_sigma1(cf_a):
    return round(np.exp(cf_a.VB_span_end_predictor_linear_prior["log_sigma1"]),3)

def return_etaKL(cf_a):
    return cf_a.eta_KL

def return_initialization_index(cf_a):
    return_initialization_index.initialization_aux += 1
    return return_initialization_index.initialization_aux 
return_initialization_index.initialization_aux = 0

def order_remove_duplicates(Selected_cf_a_list,Selected_training_logger_list, param_func):
    keys_chart = []
    aux_indexes = []
    Nselected_models = len(Selected_cf_a_list)
    print (Nselected_models)
    for i in range(Nselected_models):
        cf_a = Selected_cf_a_list[i]
        new_key = param_func(cf_a)
        print (new_key)
        if (new_key in keys_chart):  # Duplicated 
            i = i
        else:
            keys_chart.append(new_key)
            aux_indexes.append(i)
            
    print ("keys_chart: ", keys_chart)
    sorted_list, order_list = ul.sort_and_get_order(keys_chart)    
    Selected_cf_a_list_aux = []
    Selected_training_logger_list_aux = []
    for i in range (len(order_list)):
        Selected_training_logger_list_aux.append(Selected_training_logger_list[aux_indexes[order_list[i]]])
        Selected_cf_a_list_aux.append(Selected_cf_a_list[aux_indexes[order_list[i]]])
    
    Selected_training_logger_list = Selected_training_logger_list_aux
    Selected_cf_a_list = Selected_cf_a_list_aux
    return Selected_cf_a_list,Selected_training_logger_list

def return_lr(cf_a):
    return cf_a.optimizer_params["lr"]
def return_ELMoDOr(cf_a):
    return cf_a.ELMO_droput
def return_batch_size(cf_a):
    return cf_a.batch_size_train
def return_lazy_loading_size(cf_a):
    if (cf_a.datareader_lazy == True):
        return cf_a.max_instances_in_memory
    else:
        return 80000
    return cf_a.batch_size_train
def return_layers_dropout(cf_a):
#    return np.mean([cf_a.span_end_encoder_dropout,
#                    cf_a.phrase_layer_dropout,
#                    cf_a.modeling_passage_dropout,
#                    cf_a.spans_output_dropout])

        return cf_a.spans_output_dropout
    

def return_betas(cf_a):
    return cf_a.optimizer_params["betas"][0]


def return_initialization_index(cf_a):
    return_initialization_index.initialization_aux += 1
    return return_initialization_index.initialization_aux 
return_initialization_index.initialization_aux = 0

