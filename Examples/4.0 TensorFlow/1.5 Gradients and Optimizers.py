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
import tf_utilities as tful
import tf_graphic_utilities as tfgrul

plt.close("all")
tb_folder = "../TensorBoard/Examples/2_basicLR_TensBoard/"
folder_images = "../pics/Trapying/TensorBoard_Examples/"
ul.create_folder_if_needed(tb_folder)
ul.create_folder_if_needed(folder_images)

##########################################################################
################ Load the Data ########################################
##########################################################################

X_data_tr =  np.array([1.2,1.8,3.2,3.9,4.7,6.4, 4.4])
Y_data_tr =  np.array([0, -1,-2,-3,-4, -5, -9])

X_data_val =  np.array([8,9,10,11])
Y_data_val =  np.array([-7.1, -7.9,-9.2,-10.1])

#### Hyperparameters ####
lr = 0.008 # 0.0075 0.00
delta_hubber = 1.0
##########################################################################
################ CREATE THE GRAPH ########################################
##########################################################################

tf.reset_default_graph()

with tf.name_scope("Input_Data"):
    x = tf.placeholder(tf.float32) # shape = [0]
    y = tf.placeholder(tf.float32)


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

    with tf.name_scope("Network_Operations"):
        # We can use normal operators over tensors or the methods in tensorflow
        o = tf.multiply(W,x) + b

# Loss funciton is the sum of squared error.
with tf.name_scope("Loss_function"):
    loss = tful.huber_loss(y,o,delta_hubber)
#    loss = tf.reduce_sum(tf.square(y-o), name = "MSE_loss")
    
with tf.variable_scope('performance'):
    RMSE = tf.sqrt(tf.reduce_sum(tf.square(y-o), name = "MSE_performance"))
    # Tensorboars.
    accuracy = tf.summary.scalar("performance/accuracy", RMSE)


with tf.name_scope("Optimizer"):
    # First we create the optimizer object.
    optimizer = tf.train.GradientDescentOptimizer(lr)
    # Then we indicate the variable to minimize.
    train = optimizer.minimize(loss)

with tf.name_scope("Initializer"):
    init = tf.global_variables_initializer()

##########################################################################
################ CREATE A SESSION AND RUN DATA ###########################
##########################################################################

## Create a Session object to lauch the graph
sess = tf.Session();
## The graph can be referenced with 
graph = sess.graph

# Run the session to initialize all variables of the model.
sess.run(init)
# For now on, when we access the variables we will obtain their current value.

# TensorBoard to monitor Graph and training
File_Tensorboard = tf.summary.FileWriter(tb_folder, graph) 
## This will just print the Tensorflow description
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
    
    
    
# Get the last parameters
W_values,b_values = sess.run([W,b])

# Get the 
sess.close()

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
ax1 = gl.plot([], tr_loss, nf = 1, lw = 3, labels = ["RMSE loss and parameters. Learning rate: %.3f"%lr, "","RMSE"], legend = ["train"])
gl.plot([], val_loss, lw = 3, legend = ["validation"])

ax2 = gl.plot([], W_list, nf = 1, lw = 3, sharex = ax1, labels = ["", "","Parameters"], legend = ["W"],
              color ="b")
gl.plot([], b_list, lw = 3, labels = ["", "epochs","Parameters"], legend = ["b"],color ="g")

gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 20, ylabel = 20, 
                  legend = 20, xticks = 12, yticks = 12)
    
gl.savefig(folder_images +'Training_Example_Parameters.png', 
           dpi = 100, sizeInches = [14, 7])
    


