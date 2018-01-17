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

tb_folder = "../TensorBoard/Examples/1_basic/"
ul.create_folder_if_needed(tb_folder)

## Destroy the graph
tf.reset_default_graph()

## Build the graph

## We need a global initializer

"""
Parameters of the model.
    - Described as tf.Variables() that can be trained.
    - We can give them an initial value or use an initializer.
    - The first time is accessed, the variable is created and 
     after that, it is the same variable, which can be modified
     by TF.

"""


W = tf.Variable(initial_value = [0.1], # Initialization, it knows rank 1
                trainable = True,     # Can be trained by tf
                dtype = tf.float32,   # Type of the variable
                name = "Weights")    # Name of the Variable

b = tf.Variable(initial_value = [0.4], 
                trainable = True, 
                dtype = tf.float32,
                name = "bias")

"""
Variables to feed:
    These are the variables that we will give to the graph to be executed.
    In this case they are:
        - Input values X
        - Target values y
        
    They are represented by placeholders, tensors where we do not specify 
    the values, just the dimensions
"""

x = tf.placeholder(tf.float32) # shape = [0]
y = tf.placeholder(tf.float32)

"""
Define the operations to get the output

"""
# We can use normal operators over tensors
o = tf.multiply(W,x) + b

"""
Define the loss function
"""

loss = tf.reduce_sum(tf.square(y-o))

"""
Define the optimizer.
Everything is trained by Backprogation
"""

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

"""
Finally we need to initialize the variables ??
TODO: why we need to initialize it ?
"""
init = tf.global_variables_initializer()

"""
Run the session
"""
## Create a Session object to lauch the graph
sess = tf.Session();

## The graph is stores in
graph = sess.graph

# TensorBoard to monitor Graph and training
File_Tensorboard = tf.summary.FileWriter(tb_folder, graph) 
## This will just print the Tensorflow description
"""
When we fetch we indicate first the "key" with which we will retrieve later
and then the Tensor variable that we want to fetch.

In the feed we indicate as "key" the placeholder we are feeding, and as value
the values of the placeholder
"""
fetch_dict = {
    "loss": loss,
}

feed_dict = {
        x: [1,2,3,4],
        y: [0, -1,-2,-3]
}

## Launch the session fetching different data. We could fetch more than 1 at a time

"""
Just initialize the model and obtain the loss function for some data
"""
sess.run(init)
loss_value = sess.run(fetch_dict, feed_dict)["loss"]
print ("The loss value is:" , loss_value)

"""
TRAIN THE MODEL
If the fetch the optimizer, it will train !!
"""

for i in range(1000):
    sess.run(train, feed_dict)
    
"""
See the results
"""
loss_value = sess.run(fetch_dict, feed_dict)["loss"]
print ("The loss value after training is:" , loss_value)

# We can fetch the value of the tf.Variables by fetching them

W_values,b_values = sess.run([W,b])

print ("Weight value: ", W_values)
print ("Bias value: ", b_values)

# If we print the Tensor directly we only print the reference, not the data.
print ("Weight Tensor: ", W)
print ("Bias Tensor: ", b)
# Close the session to liberate resources.
sess.close()