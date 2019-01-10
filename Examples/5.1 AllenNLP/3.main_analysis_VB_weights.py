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
import Variational_inferences_lib as Vil
from other_func_bidaf import *
plt.close("all")
"""
#################### EXECUTION FLAGS ####################
"""
Experiments_instantiate_model = 0

## Folder
folder_images = "../pics/Thesis/Eval_weights_Bayesian/"
images_prefix = "Bayesian_noRegularization_" # We add this prefix to the image names to avoid collision
ul.create_folder_if_needed(folder_images)

"""
#################### MODEL TO LOAD ####################
Information to load the experiments
"""
#source_path = "../results/bidaf/"
#model_id = "2018-12-08 18:49:45.993498" 
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

Standard_values_alyses = [0.0, 100, 0.000, 0.5,"whatever"]

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
[cf_a,training_logger] = pkl.load_pickle(pickle_results_path)
cf_a.dtype = dtype  # Variable types
cf_a.device = device

## Modify these parameters so that I am not fucked in memory in my litle servergb
cf_a.datareader_lazy = True # Force lazyness for RAM optimization
cf_a.batch_size_train = 30
cf_a.batch_size_validation = 30 
cf_a.force_free_batch_memory = False
max_instances_in_memory = 1000
print_conf_params(cf_a)

folder_images+= "Eta_"+str(cf_a.eta_KL) + "_DOr_" + str(cf_a.spans_output_dropout) + \
"_sigma_" + str(round(np.exp(cf_a.VB_span_end_predictor_linear_prior["log_sigma1"]),3))
"""
##################################################################
############ INSTANTIATE DATAREADER AND LOAD DATASET ############
##################################################################
"""
vocab = Vocabulary()

"""
########################################################
################# INSTANTIATE THE MODEL ###################
"""
if (Experiments_instantiate_model):
    print ("Initializing Model architecture")
    model = BidirectionalAttentionFlow_1(vocab, cf_a)
    print("Loading previous model")
    model.load_state_dict(torch.load(model_file_path))

#model.trim_model(4)

def plots_weights_layer(mu_W, sigma_W, mu_b, sigma_b, ax1, title,  
                        legend = ["Weights and biases"],alpha =  0.2):
        """
        Plot the given weights of the layer
        """
        # For each of the weights we plot them !!
        color = gl.get_color(None) 

        gl.scatter( mu_W, sigma_W, ax = ax1, labels = [title,r"$\mu_w$",r"$\sigma_w$"], 
                   color = color, legend = legend, alpha = alpha)
        gl.scatter(mu_b, sigma_b, ax = ax1,  color = color, marker = "s", alpha =alpha)
            

def get_LinearVB_weights(VBmodel):
    sigma_W = Vil.softplus(VBmodel.rho_weight).detach().cpu().numpy().flatten()
    mu_W = VBmodel.mu_weight.detach().cpu().numpy().flatten()
    
    sigma_b = Vil.softplus(VBmodel.rho_bias).detach().cpu().numpy().flatten()
    mu_b = VBmodel.mu_bias.detach().cpu().numpy().flatten()
    return [mu_W, sigma_W, mu_b, sigma_b]

def get_boundaries_plot(mu_W, sigma_W,mu_b, sigma_b):
        max_mu = np.max([np.max(mu_W),np.max(mu_b)])
        min_mu = np.min([np.min(mu_W),np.min(mu_b)])
        
        max_std = np.max([np.max(sigma_W),np.max(sigma_b)])
        min_std = np.min([np.min(sigma_W),np.min(sigma_b)])
        
        max_abs = np.max(np.abs([max_mu,min_mu]))
        
        return max_mu,min_mu,max_std,min_std,max_abs

def plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs):
        ## For the not significant region
        mu_grid = np.array([-max_abs *10,0,max_abs*10])
        y_grid = np.abs(mu_grid)/2
        gl.fill_between(mu_grid, 10*np.ones(mu_grid.size), y_grid,
                        alpha = 0.2, color = "r", ax = ax1, legend = ["95% non-significant"])
        
def plot_VB_weights_mu_std_2D(VBmodel, ax1, type = "LinearVB", title = ""):
    """
    This function plots the variational weights in the 2 axes given
    """
    l = 0
    if (type == "LinearVB"):
        [mu_W, sigma_W, mu_b, sigma_b] = get_LinearVB_weights(VBmodel)
        shape_weights = VBmodel.mu_weight.detach().cpu().numpy().shape
        title +=" " + str(shape_weights)
#        title = ["linear layer: %i"%(l)]
        
        plots_weights_layer(mu_W, sigma_W, mu_b, sigma_b, ax1, title)
        
        prior = VBmodel.prior
        max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        
        gl.scatter(0, prior.sigma1, lw = 3, ax = ax1, legend = ["Prior 1 (%.3f)"%(prior.sigma1)], color = "k", marker = "x",)
        gl.scatter(0, prior.sigma2, lw = 3,ax = ax1, legend = ["Prior 2 (%.3f)"%(prior.sigma2)], color = "b",marker = "x" ) 

        plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
        gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                     X = np.array([min_mu,max_mu]), 
                     Y = np.array([min_std,max_std]) )

    if (type == "HighwayVB"):
        [mu_W, sigma_W, mu_b, sigma_b] = get_LinearVB_weights(VBmodel)
        shape_weights = VBmodel.mu_weight.detach().cpu().numpy().shape
        title +=" " + str(shape_weights)
#        title = ["linear layer: %i"%(l)]
        print(mu_W.shape, mu_b.shape)
#        plots_weights_layer(mu_W[:200,:], sigma_W[:200,:], mu_b[:200], sigma_b[:200], ax1, title + " $G(x)$")
#        plots_weights_layer(mu_W[200:,:], sigma_W[200:,:], mu_b[200:], sigma_b[200:], ax1, title+ " $H(x)$")
      
        plots_weights_layer(mu_W.reshape(shape_weights)[:200,:].flatten(), sigma_W.reshape(shape_weights)[:200,:].flatten(), mu_b[:200], sigma_b[:200], ax1, title, legend = ["Weights and biases G(x)"])
        plots_weights_layer(mu_W.reshape(shape_weights)[200:,:].flatten(), sigma_W.reshape(shape_weights)[200:,:].flatten(), mu_b[200:], sigma_b[200:], ax1, title, legend = ["Weights and biases H(x)"])
        
        prior = VBmodel.prior
        max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        
        gl.scatter(0, prior.sigma1, lw = 3, ax = ax1, legend = ["Prior 1 (%.3f)"%(prior.sigma1)], color = "k", marker = "x",)
        gl.scatter(0, prior.sigma2, lw = 3,ax = ax1, legend = ["Prior 2 (%.3f)"%(prior.sigma2)], color = "b",marker = "x" ) 

        plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
        gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                     X = np.array([min_mu,max_mu]), 
                     Y = np.array([min_std,max_std]) )
        
def compute_all_boundaries(list_all_weights):
    all_boundaries = [[],[],[],[],[]]
    for i in range(len(list_all_weights)):
        [mu_W, sigma_W, mu_b, sigma_b] = list_all_weights [i]
#        max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        list_boundaries = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        for j in range(len(list_boundaries)):
            all_boundaries[j].append(list_boundaries[j])
    
    max_mu = np.max(all_boundaries[0])
    min_mu = np.min(all_boundaries[1])
    max_std = np.max(all_boundaries[2])
    min_std = np.min(all_boundaries[3])
    max_abs = np.max(all_boundaries[4])
    
    return max_mu,min_mu,max_std,min_std,max_abs

plot_charts = 1
if (plot_charts):
    if (True):
        Nrows = 2
        Ncols = 3
        
    list_all_axes = []
    
    gl.init_figure()
            
    list_all_weights = []
    list_all_labels = []
    if (cf_a.VB_Linear_projection_ELMO):
    #        gl.colorIndex = 0
            VBmodel = model._linear_projection_ELMO
            ax1 = gl.subplot2grid((Nrows,Ncols), (0,0), rowspan=1, colspan=1)
            plot_VB_weights_mu_std_2D(VBmodel,ax1,"LinearVB", "Linear proyection ELMo $W_{E}$")
            
            list_all_axes.append(ax1)
            list_all_labels.append("$W_{E}$")
            list_all_weights.append(get_LinearVB_weights(VBmodel))
            
    if (cf_a.VB_highway_layers):
    #        gl.colorIndex = 0
            VBmodel = model._highway_layer._module.VBmodels[0]
            ax1 = gl.subplot2grid((Nrows,Ncols), (0,1), rowspan=1, colspan=1)
            plot_VB_weights_mu_std_2D(VBmodel,ax1,"HighwayVB", "Highway matrix weights $W_{H}$")
            
            list_all_axes.append(ax1)
            list_all_labels.append("$W_{H}$")
            list_all_weights.append(get_LinearVB_weights(VBmodel))
            
    if (cf_a.VB_similarity_function):
    #        gl.colorIndex = 0
            VBmodel = model._matrix_attention._similarity_function
            ax1 = gl.subplot2grid((Nrows,Ncols), (0,2), rowspan=1, colspan=1)
            plot_VB_weights_mu_std_2D(VBmodel,ax1,"LinearVB", "Similarity matrix weights $W_{S}$")
            
            list_all_axes.append(ax1)
            list_all_labels.append("$W_{S}$")
            list_all_weights.append(get_LinearVB_weights(VBmodel))
    
    if (cf_a.VB_span_start_predictor_linear):
    #        gl.colorIndex = 0
            VBmodel = model._span_start_predictor_linear
            ax1 = gl.subplot2grid((Nrows,Ncols), (1,0), rowspan=1, colspan=1)
            plot_VB_weights_mu_std_2D(VBmodel,ax1,"LinearVB", "Output layer span start weights $W_{(p^1)}$")
            
            list_all_axes.append(ax1)
            list_all_labels.append("$W_{(p^1)}$")
            list_all_weights.append(get_LinearVB_weights(VBmodel))
            
    if (cf_a.VB_span_end_predictor_linear):
    #        gl.colorIndex = 0
            VBmodel = model._span_end_predictor_linear
            ax1 = gl.subplot2grid((Nrows,Ncols), (1,1), rowspan=1, colspan=1)
            plot_VB_weights_mu_std_2D(VBmodel,ax1,"LinearVB", "Output layer span end weights $W_{(p^2)}$")
            list_all_axes.append(ax1)
    
            list_all_axes.append(ax1)
            list_all_labels.append("$W_{(p^2)}$")
            list_all_weights.append(get_LinearVB_weights(VBmodel))
            
    ## Print all weights !! 
    gl.colorIndex = 0
    ax1 = gl.subplot2grid((Nrows,Ncols), (1,2), rowspan=1, colspan=1)
    for i in range (len(list_all_weights)):
    
        [mu_W, sigma_W, mu_b, sigma_b] = list_all_weights [i]
        legend = list_all_labels[i]
        if(i == 1):
            shape_weights = (400,200)
            plots_weights_layer(mu_W.reshape(shape_weights)[:200,:].flatten(), sigma_W.reshape(shape_weights)[:200,:].flatten(), mu_b[:200], sigma_b[:200], ax1, "All weights",  legend = [legend + " G(x)"], alpha = 0.1)
            plots_weights_layer(mu_W.reshape(shape_weights)[200:,:].flatten(), sigma_W.reshape(shape_weights)[200:,:].flatten(), mu_b[200:], sigma_b[200:], ax1, "All weights",  legend = [legend + " H(x)"], alpha = 0.1)
        else:
            plots_weights_layer(mu_W, sigma_W,mu_b, sigma_b, ax1, "All weights",  legend = [legend], alpha = 0.1)
        list_all_axes.append(ax1)
    
    max_mu,min_mu,max_std,min_std,max_abs = compute_all_boundaries(list_all_weights)
    plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
    gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                 X = np.array([min_mu,max_mu]), 
                 Y = np.array([min_std,max_std]) )
    
    gl.set_fontSizes(ax = list_all_axes, title = 15, xlabel = 15, ylabel = 15, 
                  legend = 10, xticks = 12, yticks = 12)
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.22)
    for ax in list_all_axes:
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    gl.savefig(folder_images +'Bayesian_weights_biday.png', 
               dpi = 100, sizeInches = [18, 12])
    
    
    
    if(1):
        
        """
        PLOT OF THE SPAN INPUT WEIGHTS BY TYPE OF THE WEIGHTS !!
        """
        
        gl.init_figure()
        
        labels_weights = [r"$H$",r"$\tilde{U}$",
                          r"$H \circ \tilde{U}$", r"$H \circ \tilde{H}$",r"$M$"]
        
        [mu_W, sigma_W, mu_b, sigma_b] = list_all_weights [-2]
        
        ax1 = None
        list_all_axes = []
        for i in range(len(labels_weights)):
            ax1 = gl.subplot2grid((2,3), (int(i/3),i%3), rowspan=1, colspan=1,sharex = ax1, sharey = ax1) 
            plots_weights_layer(mu_W[200*i:200*(i+1)], 
                               sigma_W[200*i:200*(i+1)],
                               sigma_b, 
                               mu_b, 
                               ax1,"Span end weights $W_{(p^1)}$" ,  legend = ["Weight for " + labels_weights[i]])
            
            max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
            plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
            gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                         X = np.array([min_mu,max_mu]), 
                         Y = np.array([min_std,max_std]) )
    
            list_all_axes.append(ax1)
            
        gl.colorIndex = 0
        ax1 = gl.subplot2grid((2,3), (1,2), rowspan=1, colspan=1,sharex = ax1, sharey = ax1)
        for i in range(len(labels_weights)):
           plots_weights_layer(mu_W[200*i:200*(i+1)], 
                               sigma_W[200*i:200*(i+1)],
                               sigma_b, 
                               mu_b, 
                               ax1,"Span end weights $W_{(p^1)}$",  legend = ["Weights for " + labels_weights[i]])
           
        max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
        gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                     X = np.array([min_mu,max_mu]), 
                     Y = np.array([min_std,max_std]) )
            
        list_all_axes.append(ax1)
        gl.set_fontSizes(ax = list_all_axes, title = 15, xlabel = 15, ylabel = 15, 
                      legend = 12, xticks = 12, yticks = 12)
        
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.22)
        for ax in list_all_axes:
            ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
        gl.savefig(folder_images +'Span_p1_weights_VB.png', 
                   dpi = 100, sizeInches = [18, 12])
        
        """
        PLOT OF THE SPAN INPUT WEIGHTS BY TYPE OF THE WEIGHTS !!
        """
        
        gl.init_figure()
        
        labels_weights = [r"$H$",r"$\tilde{U}$",
                          r"$H \circ \tilde{U}$", r"$H \circ \tilde{H}$",r"$M^2$"]
        
        [mu_W, sigma_W, mu_b, sigma_b] = list_all_weights [-1]
        
        ax1 = None
        list_all_axes = []
        for i in range(len(labels_weights)):
            ax1 = gl.subplot2grid((2,3), (int(i/3),i%3), rowspan=1, colspan=1,sharex = ax1, sharey = ax1) 
            plots_weights_layer(mu_W[200*i:200*(i+1)], 
                               sigma_W[200*i:200*(i+1)],
                               sigma_b, 
                               mu_b, 
                               ax1,"Span start weights $W_{(p^2)}$" ,  legend = ["Weight for " + labels_weights[i]])
            
            max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
            plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
            gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                         X = np.array([min_mu,max_mu]), 
                         Y = np.array([min_std,max_std]) )
            
            list_all_axes.append(ax1)
        gl.colorIndex = 0
        ax1 = gl.subplot2grid((2,3), (1,2), rowspan=1, colspan=1,sharex = ax1, sharey = ax1)
        for i in range(len(labels_weights)):
           plots_weights_layer(mu_W[200*i:200*(i+1)], 
                               sigma_W[200*i:200*(i+1)],
                               sigma_b, 
                               mu_b, 
                               ax1,"Span start weights $W_{(p^2)}$",  legend = ["Weights for " + labels_weights[i]])
        max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
        gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                     X = np.array([min_mu,max_mu]), 
                     Y = np.array([min_std,max_std]) )
        list_all_axes.append(ax1)
        gl.set_fontSizes(ax = list_all_axes, title = 15, xlabel = 15, ylabel = 15, 
                      legend = 12, xticks = 12, yticks = 12)
        
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.22)
        for ax in list_all_axes:
            ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
        gl.savefig(folder_images +'Span_p2_weights_VB.png', 
                   dpi = 100, sizeInches = [18, 12])
        
        """
        PLOT OF THE ATTENTION MATRIX WEIGHTS !!!
        """
        
        gl.init_figure()
        
        labels_weights = [r"$H$",r"$U$",
                          r"$H \circ U$"]
        
        [mu_W, sigma_W, mu_b, sigma_b] = list_all_weights [-3]
        
        ax1 = None
        list_all_axes = []
        for i in range(len(labels_weights)):
            ax1 = gl.subplot2grid((2,2), (int(i/2),i%2), rowspan=1, colspan=1,sharex = ax1, sharey = ax1) 
            plots_weights_layer(mu_W[200*i:200*(i+1)], 
                               sigma_W[200*i:200*(i+1)],
                               sigma_b, 
                               mu_b, 
                               ax1,"Attention weights $W_{(S)}$" ,  legend = ["Weight for " + labels_weights[i]])
            max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
            plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
            gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                         X = np.array([min_mu,max_mu]), 
                         Y = np.array([min_std,max_std]) )
            list_all_axes.append(ax1)
        gl.colorIndex = 0
        ax1 = gl.subplot2grid((2,2), (1,1), rowspan=1, colspan=1,sharex = ax1, sharey = ax1)
        for i in range(len(labels_weights)):
           plots_weights_layer(mu_W[200*i:200*(i+1)], 
                               sigma_W[200*i:200*(i+1)],
                               sigma_b, 
                               mu_b, 
                               ax1,"Attention weights $W_{(S)}$",  legend = ["Weights for " + labels_weights[i]])
        max_mu,min_mu,max_std,min_std,max_abs = get_boundaries_plot(mu_W, sigma_W, mu_b,sigma_b)
        plot_signifant_region(ax1, max_mu,min_mu,max_std,min_std,max_abs)
        gl.set_zoom (ax = ax1, xlimPad = [0.1, 0.1], ylimPad = [0.1,0.1], 
                     X = np.array([min_mu,max_mu]), 
                     Y = np.array([min_std,max_std]) )
        list_all_axes.append(ax1)
        gl.set_fontSizes(ax = list_all_axes, title = 15, xlabel = 15, ylabel = 15, 
                      legend = 12, xticks = 12, yticks = 12)
        
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.22)
        for ax in list_all_axes:
            ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
        gl.savefig(folder_images +'Span_attention_weights_VB.png', 
                   dpi = 100, sizeInches = [18, 12])
        
    
    
    """
    PLOT OF THE SIGNIFICANCE OF THE WEIGHTS !!
    """
    import seaborn as sns
    def distplot_1D_seaborn(data, ax, 
                            labels = ['Distribution of Context Length',"",'Context Length']
                            , color = None):
        """ Function that generates a 1D distribution plot from the data
             and saves it to disk
        """
        color = gl.get_color(color) 
        ax = sns.distplot(data, ax = ax, hist = True, norm_hist = True, color = color)
        ax.set_title(labels[0])
        ax.set_ylabel(labels[2])
        ax.set_xlabel(labels[1])
        return ax
    
    gl.init_figure()
    sharex = None
    sharey = None
    nPLOTS = len(list_all_weights)
    
    for i in range (nPLOTS):
        ax1 = gl.subplot2grid((nPLOTS,1), (i,0), rowspan=1, colspan=1, sharex = sharex, sharey = sharey)
        [mu_W, sigma_W, mu_b, sigma_b] = list_all_weights [i]
        
        ## If we want them all in the same order
        sharex = ax1
    #    sharey = ax1
        
        all_mu = np.concatenate((mu_W,mu_b))
        all_sigmae = np.concatenate((sigma_W,sigma_b))
        
        ratio = np.abs(all_mu)/all_sigmae
        
        legend = list_all_labels[i]
        
        if(i == 0):
            title = "Distribution of $|\mu_w|/\sigma_w$ for the Variational weights"
            labels = [title, "",  list_all_labels[i]]
        elif (i == nPLOTS - 1):
             labels = ["", "$|\mu_w|/\sigma_w$",  list_all_labels[i]]
        else:
            labels = ["", "",  list_all_labels[i]]
            
        distplot_1D_seaborn(ratio, ax1, labels =labels)
    
    gl.set_fontSizes(ax = list_all_axes, title = 18, xlabel = 15, ylabel = 20, 
                  legend = 10, xticks = 12, yticks = 12)
    gl.set_zoom(ax = ax1, xlim = [-0.5, np.std(ratio)*2])
    
    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.0)
    
    gl.savefig(folder_images +'Bayesian_weights_significance.png', 
               dpi = 100, sizeInches = [10, 4])
