"""
In this document we will perform a basic Linear Regression including TensorBoard.
It is a very simple manual program with no proper naming of the variables.
We will:
    - Use the conditional in the huber loss.
    - We can also save the model parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.chdir("../../")
import import_folders

import numpy as np
## Import standard libraries
import tensorflow as tf
# Import libraries

from graph_lib import gl
import numpy as np
import matplotlib.pyplot as plt

import utilities_lib as ul
import tf_general_model1 as tfm1

plt.close("all")
tb_folder = "../TensorBoard/Examples/3_LR_copmlete_Model/"
variables_folder = "../TensorBoard/Examples/3_LR_copmlete_Model/Variables/"
folder_images = "../pics/Trapying/TensorBoard_Examples/"
ul.create_folder_if_needed(tb_folder)
ul.create_folder_if_needed(variables_folder)
ul.create_folder_if_needed(folder_images)

##########################################################################
################ Load the Data ########################################
##########################################################################

X_data_tr =  np.array([1.2,1.8,3.2,3.9,4.7,6.4, 4.4])
Y_data_tr =  np.array([0, -1,-2,-3,-4, -5, -9])

X_data_val =  np.array([8,9,10,11])
Y_data_val =  np.array([-7.1, -7.9,-9.2,-10.1])

#### Hyperparameters ####

class loss_config(object):
    name = "Hubber"   # "MSE"
    delta = 1.0
    
class train_config(object):
    name = "Ada"
    lr = 0.008 
    
    saver = True;
    variables_path = variables_folder;
    
##########################################################################
################ CREATE THE GRAPH ########################################
##########################################################################

tf.reset_default_graph()

Network = tfm1.General_Model()
Network.build_graph(loss_config, train_config)

################################################################################
################ CREATE A SESSION AND RUN DATA ###########################
##########################################################################

restored_flag = Network.restore_variables();
if (restored_flag == False):
    ## If there were no checkpoints to be restored
    Network.initialize();

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

#################################################################################
################ GET THE INITIAL VARIABLES AND LOSSES  ###########################
################################################################################
## Data structures for plotting this specific trainng
tr_loss = [];
val_loss = [];
W_list = [];
b_list = []

## Add the initialization parameters
W_values,b_values,s_value = Network.get_variables()
W_list.append(W_values)
b_list.append(b_values)
# We use two summary writers. This is a hack that allows us to write 
# show two plots in the same fiigure in TensorBoard

### Propagate train and test data !!!! #####
loss_tr, accuracy_tr = Network.get_loss_accuracy(X_data_tr, Y_data_tr)
loss_val, accuracy_val  = Network.get_loss_accuracy(X_data_val, Y_data_val)
## Alternatively we can just obtain the values and plot them ourselves
tr_loss.append(loss_tr)
val_loss.append(loss_val)

#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:

print ("The initial loss value is:" , loss_tr)
print ("The variables:" , W_values,b_values,s_value)

#################################################################################
################ GET THE INITIAL VARIABLES AND LOSSES  ###########################
################################################################################

step_period = 3;
Num_iterations = 108
initial_global_step = Network.get_global_step()

if (restored_flag == False):
    ## Save the initializations
    Network.save_variables()
    
for step in range(Num_iterations):

    ## Fetch the trainer to train the network
    Network.run_epoch(X_data_tr, Y_data_tr)
    
    ## Now we fetch just 
    ### Propagate train and test data !!!! #####
    loss_tr, accuracy_tr = Network.get_loss_accuracy(X_data_tr, Y_data_tr)
    loss_val, accuracy_val  = Network.get_loss_accuracy(X_data_val, Y_data_val)
    ## Alternatively we can just obtain the values and plot them ourselves
    tr_loss.append(loss_tr)
    val_loss.append(loss_val)

    ############## SAVE THE HISTOGRAM OF THE PARAMETERS #############
    W_values,b_values,s_value = Network.get_variables()
    W_list.append(W_values)
    b_list.append(b_values)
    
    if ((step+1) % step_period == 0):
        # This will always happen after the first Num_iterations
        Network.save_variables()

loss_tr, accuracy_tr = Network.get_loss_accuracy(X_data_tr, Y_data_tr)
print ("The final loss:" , loss_tr)
print ("The final variables:" , W_values,b_values,s_value)

## Plot the loss function against the parameters !! 
## Get the surface for the loss


####### PLOT THE EVOLUTION OF RMSE AND PARAMETERS ############
gl.init_figure()
ax1 = gl.scatter(X_data_tr, Y_data_tr, lw = 3,legend = ["tr points"], labels = ["Data", "X","Y"])
ax2 = gl.scatter(X_data_val, Y_data_val, lw = 3,legend = ["val points"])

gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 20, xticks = 12, yticks = 12)

x_grid = np.linspace(np.min([X_data_tr]) -1, np.max([X_data_val]) +1, 100)
y_grid = x_grid * W_values + b_values

gl.plot (x_grid, y_grid, legend = ["training line"])
gl.savefig(folder_images +'Training_Example_Data.png', 
           dpi = 100, sizeInches = [14, 4])

####### PLOT THE EVOLUTION OF RMSE AND PARAMETERS ############
gl.set_subplots(2,1)
ax1 = gl.plot([], tr_loss, nf = 1, lw = 3, labels = ["RMSE loss and parameters. Learning rate: %.3f"%train_config.lr, "","RMSE"], legend = ["train"])
gl.plot([], val_loss, lw = 3, legend = ["validation"])

ax2 = gl.plot([], W_list, nf = 1, lw = 3, sharex = ax1, labels = ["", "","Parameters"], legend = ["W"],
              color ="b")
gl.plot([], b_list, lw = 3, labels = ["", "epochs","Parameters"], legend = ["b"],color ="g")

gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 20, xticks = 12, yticks = 12)
    
gl.savefig(folder_images +'Training_Example_Parameters.png', 
           dpi = 100, sizeInches = [14, 7])
    


#######################################################################################################
###################### CODE TO RESTORE THE VALUE OF THE CHECKPOINTS ###############################
######################################################################################################
plot_from_checkpoints = 1
if (plot_from_checkpoints):
    tr_loss = [];
    val_loss = [];
    W_list = [];
    b_list = []
    
    import tf_checkpoints_utils as tfchul
    caca = tfchul.get_all_checkpoints_paths(variables_folder)
    gss = tfchul.get_all_checkpoints_gloabl_step(caca)
    
    for model_checkpoint in caca:
        Network.restore_variables(model_checkpoint);
    
    
        ## Data structures
        Network.run_epoch(X_data_tr, Y_data_tr)
        
        ## Now we fetch just 
        ### Propagate train and test data !!!! #####
        loss_tr, accuracy_tr = Network.get_loss_accuracy(X_data_tr, Y_data_tr)
        loss_val, accuracy_val  = Network.get_loss_accuracy(X_data_val, Y_data_val)
        ## Alternatively we can just obtain the values and plot them ourselves
        tr_loss.append(loss_tr)
        val_loss.append(loss_val)
        
        ############## SAVE THE HISTOGRAM OF THE PARAMETERS #############
        W_values,b_values,s_value = Network.get_variables()
        W_list.append(W_values)
        b_list.append(b_values)
        
        if (step % step_period == 0):
            Network.save_variables()
    
    
    
    ####### PLOT THE EVOLUTION OF RMSE AND PARAMETERS ############
    gl.set_subplots(2,1)
    ax1 = gl.plot(gss, tr_loss, nf = 1, lw = 3, labels = ["RMSE loss and parameters. Learning rate: %.3f"%train_config.lr, "","RMSE"], legend = ["train"])
    gl.plot(gss, val_loss, lw = 3, legend = ["validation"])
    
    ax2 = gl.plot(gss, W_list, nf = 1, lw = 3, sharex = ax1, labels = ["", "","Parameters"], legend = ["W"],
                  color ="b")
    gl.plot(gss, b_list, lw = 3, labels = ["", "epochs","Parameters"], legend = ["b"],color ="g")
    
    gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                      legend = 20, xticks = 12, yticks = 12)
        
    gl.savefig(folder_images +'Training_Example_Parameters.png', 
               dpi = 100, sizeInches = [14, 7])
    

# Get the 
Network.close_session()