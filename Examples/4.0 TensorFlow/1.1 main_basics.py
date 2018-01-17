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
a = tf.constant(5.0)
b = tf.constant(6.9)

c = tf.multiply(a,b)
# We can use normal operators over tensors
d = a * c 

## Create a Session object to lauch the graph
sess = tf.Session();

## The graph is stores in
graph = sess.graph

# TensorBoard to monitor Graph and training

File_Tensorboard = tf.summary.FileWriter(tb_folder, graph) 
## This will just print the Tensorflow description
print ("Printing the Tensor properties, no session run", c)

## Launch the session fetching different data. We could fetch more than 1 at a time

c_value = sess.run(c)
d_value = sess.run(d)
print ("Printing the Tensor values after session run", c_value)

# Close the session to liberate resources.
sess.close()