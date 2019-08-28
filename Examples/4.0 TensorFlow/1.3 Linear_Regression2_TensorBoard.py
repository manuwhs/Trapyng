"""
In this document we will perform a basic Linear Regression including TensorBoard.
It is a very simple manual program with no proper naming of the variables.
We will:
    - Define placeholders for the input variables
    - Define the operations to compute the output.
    - Define a loss function over the output.
    - Defina an optimizer for training.

    - Regarding the saving of information from training:
        - We save in tensorBoard the train and validation accuracy
        - We save the train and validation accuracy in python.
        The advantage of tensorBoard is that if the training process fails then 
        we had the things saved on this so we can fetch the results. 
        
        - We can also save the model parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.chdir("../../")
import import_folders

import graph_lib as gl
import numpy as np
## Import standard libraries
import tensorflow as tf
# Import libraries
import subprocess
import reader

import utilities_lib as utils 
## Import TF creared 
import BBB_LSTM_configs as Bconf
import BBB_LSTM_Model as BM
import Variational_inferences_lib as VI
import util_autoparallel as util

import pickle_lib as pkl

from graph_lib import gl
import numpy as np
import matplotlib.pyplot as plt

import utilities_lib as ul
import tf_graphic_utilities as tfgrul

plt.close("all")
tb_folder = "../TensorBoard/Examples/2_basicLR_TensBoard/"
folder_images = "../pics/Trapying/TensorBoard_Examples/"
ul.create_folder_if_needed(tb_folder)
ul.create_folder_if_needed(folder_images)

##########################################################################
################ Load the Data ########################################
##########################################################################

X_data_tr =  np.array([1.2,1.8,3.2,3.9,4.7,6.4])
Y_data_tr =  np.array([0, -1,-2,-3,-4, -5])

X_data_val =  np.array([8,9,10,11])
Y_data_val =  np.array([-7.1, -7.9,-9.2,-10.1])

#### Hyperparameters ####
lr = 0.0052 # 0.0075 0.00
##########################################################################
################ CREATE THE GRAPH ########################################
##########################################################################

## Destroy the previous graph just in case
tf.reset_default_graph()
## Build the graph
#with tf.Graph().as_default():
#    
"""
1. INPUT VARIABLES OF THE MODEL
    These are the variables that we will give to the computational 
    when we want to run the session. The input variables will be processed to obtain the
    output variables that we are fetching. 
    In this case they are:
        - Input values X
        - Target values y
        
    They are represented by placeholders, tensors where we do not specify 
    the values, just the dimensions
"""
with tf.name_scope("Input_Data"):
    x = tf.placeholder(tf.float32) # shape = [0]
    y = tf.placeholder(tf.float32)

"""
2. PARAMETERS OF THE MODEL.
    There are the variables that the algorithm needs to perform the computation.
    They could be trainable or non-trainable depending on what we want.
    - Described as tf.Variables().
    - We can give them an initial value or use an initializer.
    - The first time is accessed, the variable is created and 
     after that, it is the same variable, which can be modified
     by TF.

"""

with tf.name_scope("Model"):
    with tf.name_scope("Model_Variables"):
        
        W = tf.Variable(initial_value = [-2.1], # Initialization, it knows rank 1
                        trainable = True,     # Can be trained by tf
                        dtype = tf.float32,   # Type of the variable
                        name = "Weights")    # Name of the Variable
        
        b = tf.Variable(initial_value = [-1.5], 
                        trainable = True, 
                        dtype = tf.float32,
                        name = "bias")
        
    
    
    """
    3. OPERATIONS
        Define the set of operation that are made between the tensors in order to
        generate the output of the model. In this case we just multiply input and output.
    
    """
    
    with tf.name_scope("Network_Operations"):
        # We can use normal operators over tensors or the methods in tensorflow
        o = tf.multiply(W,x) + b


# Loss funciton is the sum of squared error.
with tf.name_scope("Loss_function"):
    loss = tf.reduce_sum(tf.square(y-o), name = "MSE_loss")
with tf.variable_scope('performance'):
    RMSE = tf.sqrt(tf.reduce_sum(tf.square(y-o), name = "MSE_performance"))
    # Tensorboars.
    accuracy = tf.summary.scalar("performance/accuracy", RMSE)

"""
5. DEFINE THE OPTIMIZER.
    Everything is trained by Backprogation
"""
with tf.name_scope("Optimizer"):
    # First we create the optimizer object.
    optimizer = tf.train.GradientDescentOptimizer(lr)
    # Then we indicate the variable to minimize.
    train = optimizer.minimize(loss)
"""
6. DEFINE THE INITIALIZER.
     We need a global initializer to initilize every variable
"""
with tf.name_scope("Initializer"):
    init = tf.global_variables_initializer()

"""
Run the session
"""

##########################################################################
################ CREATE A SESSION AND RUN DATA ###########################
##########################################################################

## Create a Session object to lauch the graph
sess = tf.Session();
## The graph can be referenced with 
graph = sess.graph


"""
1. INITIALIZE VARIABLES
Finally we need to initialize the variables ??
TODO: why we need to initialize it ?
"""
## We need a global initializer

# Run the session to initialize all variables of the model.
sess.run(init)
# For now on, when we access the variables we will obtain their current value.

"""
2. TENSORBOARD
Finally we need to initialize the variables ??
TODO: why we need to initialize it ?
"""
# TensorBoard to monitor Graph and training
File_Tensorboard = tf.summary.FileWriter(tb_folder, graph) 
## This will just print the Tensorflow description

"""
When we fetch we indicate first the "key" with which we will retrieve later
and then the Tensor variable that we want to fetch.

In the feed we indicate as "key" the placeholder we are feeding, and as value
the values of the placeholder
"""
##########################################################################
################ FORWARD PASS  ###########################
##########################################################################

# Let us just get the output for the data.
fetch_dict = {
    "output": o,
}

feed_dict = {
        x: X_data_tr,
}

o_values = sess.run(fetch_dict, feed_dict)["output"]
print ("The propagated outputs are" , o_values)

########################################################
################ TRAINING  ###########################
#######################################################

## Data structures
tr_loss = [];
val_loss = [];
W_list = [];
b_list = []

## Add the initialization parameters
W_values,b_values = sess.run([W,b])
W_list.append(W_values)
b_list.append(b_values)
# We use two summary writers. This is a hack that allows us to write 
# show two plots in the same fiigure in TensorBoard
summary_writer_train = tf.summary.FileWriter(os.path.join(tb_folder, 'train'), sess.graph)
summary_writer_valid = tf.summary.FileWriter(os.path.join(tb_folder, 'valid'), sess.graph)

## Launch the session fetching different data. We could fetch more than 1 at a time
fetch_dict = {
    "loss": loss,
    "accuracy": accuracy,
    }           
## Now we fetch just 
feed_dict_tr = {
    x: X_data_tr,
    y: Y_data_tr
}
feed_dict_val = {
    x: X_data_val,
    y: Y_data_val
}

fetch_dict = {
"loss": RMSE,
"accuracy": accuracy,
}   
### Propagate train and test data !!!! #####
fetched_data_tr = sess.run(fetch_dict, feed_dict_tr)
fetched_data_val = sess.run(fetch_dict, feed_dict_val)

## Alternatively we can just obtain the values and plot them ourselves
tr_loss.append(float(fetched_data_tr["loss"]))
val_loss.append(float(fetched_data_val["loss"]))

    
feed_dict_tr = {
        x: X_data_tr,
        y: Y_data_tr
    }
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
loss_value = sess.run(fetch_dict, feed_dict_tr)["loss"]
print ("The initial loss value is:" , loss_value)

"""
TRAIN THE MODEL
If the fetch the optimizer, it will train !!
"""
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:

for i in range(300):

    ## Fetch the trainer to train the network
    sess.run(train, feed_dict_tr)
    
    ## Now we fetch just 
    feed_dict_tr = {
        x: X_data_tr,
        y: Y_data_tr
    }
    feed_dict_val = {
        x: X_data_val,
        y: Y_data_val
    }
    
    fetch_dict = {
    "loss": RMSE,
    "accuracy": accuracy,
    }   
    ### Propagate train and test data !!!! #####
    fetched_data_tr = sess.run(fetch_dict, feed_dict_tr)
    fetched_data_val = sess.run(fetch_dict, feed_dict_val)
    
    ## Call the writters to store de values in TensorBoard !!
    summary_writer_train.add_summary(fetched_data_tr["accuracy"], i)
    summary_writer_valid.add_summary(fetched_data_val["accuracy"], i)
    
    ## Alternatively we can just obtain the values and plot them ourselves
    tr_loss.append(float(fetched_data_tr["loss"]))
    val_loss.append(float(fetched_data_val["loss"]))

    ############## SAVE THE HISTOGRAM OF THE PARAMETERS #############
    W_values,b_values = sess.run([W,b])
    W_list.append(W_values)
    b_list.append(b_values)
    
    
        
"""
See the results
"""

#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
loss_value = sess.run(fetch_dict, feed_dict_tr)["loss"]
print ("The loss value after training is:" , loss_value)

# We can fetch the value of the tf.Variables by fetching them
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
W_values,b_values = sess.run([W,b])

print ("Weight value: ", W_values)
print ("Bias value: ", b_values)

# If we print the Tensor directly we only print the reference, not the data.
print ("Weight Tensor: ", W)
print ("Bias Tensor: ", b)
# Close the session to liberate resources.
sess.close()



## Plot the loss function against the parameters !! 


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
ax1 = gl.plot([], tr_loss, nf = 1, lw = 3, labels = ["RMSE loss and parameters. Learning rate: %.3f"%lr, "","RMSE"], legend = ["train"])
gl.plot([], val_loss, lw = 3, legend = ["validation"])

ax2 = gl.plot([], W_list, nf = 1, lw = 3, sharex = ax1, labels = ["", "","Parameters"], legend = ["W"],
              color ="b")
gl.plot([], b_list, lw = 3, labels = ["", "epochs","Parameters"], legend = ["b"],color ="g")

gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 20, xticks = 12, yticks = 12)
    
gl.savefig(folder_images +'Training_Example_Parameters.png', 
           dpi = 100, sizeInches = [14, 7])
    
####### PLOT THE EVOLUTION OF RMSE AND PARAMETERS in 3D ############
gl.set_subplots(1,3)
W_grid_mesh, b_grid_mesh, loss_grid_mesh = tfgrul.get_3D_surface_loss(X_data_tr, Y_data_tr, N_w = 100, N_b = 110)
ax1 = gl.plot_3D(W_grid_mesh, b_grid_mesh, loss_grid_mesh, nf = 1, alpha = 0.8, labels = ["Train and validation RMSE surfaces","W","b","RMSE"], legend = ["train"])

W_grid_mesh, b_grid_mesh, loss_grid_mesh = tfgrul.get_3D_surface_loss(X_data_val, Y_data_val, N_w = 100, N_b = 110)
gl.plot_3D(W_grid_mesh, b_grid_mesh, loss_grid_mesh, alpha = 0.8,legend = ["val"])

##### Plot the training steps ###########


W_grid_mesh, b_grid_mesh, loss_grid_mesh = tfgrul.get_3D_surface_loss(X_data_tr, Y_data_tr, N_w = 100, N_b = 110)
ax2 = gl.plot_3D(W_grid_mesh, b_grid_mesh, loss_grid_mesh, nf = 1, alpha = 0.2, sharex = ax1, sharey = ax1,
                 labels = ["Evolution of the training RMSE loss","W","b","RMSE"], legend = ["train"])
selected_W, selected_b, losses = tfgrul.get_training_points(X_data_tr, Y_data_tr, W_list, b_list, N = 150)
gl.scatter_3D(selected_W, selected_b, losses, join_points = "yes")

W_grid_mesh, b_grid_mesh, loss_grid_mesh = tfgrul.get_3D_surface_loss(X_data_val, Y_data_val, N_w = 100, N_b = 110)
ax3 = gl.plot_3D(W_grid_mesh, b_grid_mesh, loss_grid_mesh, nf = 1, alpha = 0.2, sharex = ax1, sharey = ax1,
                 labels = ["Evolution of the validation RMSE loss","W","b","RMSE"],legend = ["val"])
selected_W, selected_b, losses = tfgrul.get_training_points(X_data_val, Y_data_val, W_list, b_list, N = 150)
gl.scatter_3D(selected_W, selected_b, losses, join_points = "yes")

gl.savefig(folder_images +'Training_Example_LossFunction.png', 
           dpi = 100, sizeInches = [25, 7])

#ax1 = gl.plot([], tr_loss, nf = 1, labels = ["RMSE loss for each epoch of training. Learning rate: %.3f"%lr, "epoch","RMSE"], legend = ["train"])
#gl.plot([], val_loss, labels = ["MSE loss for each epoch of training", "epoch","Value"], legend = ["validation"])
#
#ax2 = gl.plot([], W_list, nf = 1, sharex = ax1, labels = ["Parameters", "epoch","Value"], legend = ["W"])
#gl.plot([], b_list, labels = ["Parameters", "epoch","RMSE"], legend = ["b"])
#
#gl.savefig(folder_images +'Training_Example_LossFunction.png', 
#           dpi = 100, sizeInches = [14, 7])

