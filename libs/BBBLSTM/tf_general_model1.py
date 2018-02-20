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


class General_Model:
    """
    This Model is general purpose, it provides a set of parts of the graph. 
    The different parts would be interconnected with interfaces to better assemply
    """
    def __init__(self, params):
        """
        Function to 
        """

        pass
    
    def _create_placeholders(self):
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

        
        pass
    
    def _create_IO_graph(self):
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
        """
        3. OPERATIONS
            Define the set of operation that are made between the tensors in order to
            generate the output of the model. In this case we just multiply input and output.
        
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
        with tf.name_scope("Network_Operations"):
            # We can use normal operators over tensors or the methods in tensorflow
            o = tf.multiply(W,x) + b        
                
    """
    4. DEFINE THE LOSS FUNCTION
        We will define the cost function that we want to minimize. In this case
        we want to minimize the squared error of the regression. We indicate it
        by creating a variable where we store the output of the cost function.
    """
        def _create_loss(self):
            """ Step 3 + 4: define the inference + the loss function """
            pass
    
    def _create_IO_graph(self):
    with tf.name_scope("Loss_function"):
        loss = tf.reduce_sum(tf.square(y-o), name = "MSE_loss")
    with tf.variable_scope('performance'):
        RMSE = tf.sqrt(tf.reduce_sum(tf.square(y-o), name = "MSE_performance"))
        # Tensorboars.
        accuracy = tf.summary.scalar("performance/accuracy", RMSE)



    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        pass

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        pass

    


