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
    def __init__(self):
        """
        Function to 
        """
        self.saver = None;
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

        #### Store the input variables in the model
        self.x = x;
        self.y = y;
        
        return x,y

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
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
            output = tf.multiply(W,self.x) + b        
    
    #### Store the output variables in the model
        self.output = output;
        self.variables = [W,b];
        return output
    """
    4. DEFINE THE LOSS FUNCTION
        We will define the cost function that we want to minimize. In this case
        we want to minimize the squared error of the regression. We indicate it
        by creating a variable where we store the output of the cost function.
    """
    def _create_loss(self,loss_config):
        """ Step 3 + 4: define the inference + the loss function """


        with tf.name_scope("Loss_function"):
            
            if (loss_config.name == "Hubber"):
                loss = tful.huber_loss(self.y,self.output,loss_config.delta)
                
                
            elif(loss_config.name == "MSE"):
                loss = tf.reduce_sum(tf.square(self.y-self.output), name = "MSE_loss")
                
        with tf.variable_scope('performance'):
            RMSE = tf.sqrt(tf.reduce_sum(tf.square(self.y-self.output), name = "MSE_performance"))
            # Tensorboars. Variable that we want to be saved during training so that we do not need to compute it later.
            accuracy = tf.summary.scalar("performance/accuracy", RMSE)

        self.loss = loss;
        self.accuracy = accuracy;
        
        return loss, accuracy



    def _create_saver(self, path):
        """
        Saver to automatically save:
            - Variables
            - Accuracy / Loss (for later)
            
        It can be power every certain global step or time
        """
        
        saver = tf.train.Saver()
        
        # Variable to store the number of iteration of train,  Very common practice in ternsorflow.
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.variables_path = path
        
        self.saver = saver
        
    def _create_optimizer(self, train_config):
        with tf.name_scope("Optimizer"):
            # First we create the optimizer object.
            optimizer = tf.train.GradientDescentOptimizer(train_config.lr)
            # Then we indicate the variable to minimize.
            train = optimizer.minimize(self.loss,global_step=self.global_step)
        
        self.optimizer = optimizer;
        self.train = train;
        return optimizer,train

    def _create_initializer(self):
        with tf.name_scope("Initializer"):
            initializer = tf.global_variables_initializer()

        self.initializer = initializer;
        return initializer;
    
    ##################################################################################
    ##################################################################################
    ##################################################################################

    def build_graph(self,loss_config, train_config):
        """
        Call all the builders of the graph :)
        """
        self._create_placeholders();
        self._create_embedding()
        
        self._create_IO_graph();
        self._create_loss(loss_config);
        
        if (train_config.saver):
             self._create_saver(train_config.variables_path);
             
        self._create_optimizer(train_config);
        
        self._create_initializer()
        
    def initialize(self ):
        """
        Initialize the variables of the network
        """
        
        session = tf.Session();
        self.session = session
        # Run the session to initialize all variables of the model.
        session.run(self.initializer)
        
        
    def save_variables(self):
        self.saver.save(self.session, self.variables_path, global_step=self.global_step)
        
    def get_variables(self):
        """
        Get all the variables of the network.
        We need to remember the way they were stored in the first place
        """
        variables = self.session.run(self.variables)

        return variables
    
    def get_output(self, Xdata):
        """
        Get the output of the network for some given x as input
        """
        
        fetch_dict = {
            "output": self.output,
        }        
        feed_dict = {
            self.x: Xdata
        }
        
        ### Propagate train and test data !!!! #####
        fetched_data = self.session.run(fetch_dict, feed_dict)
        
        return fetched_data["output"]
        

    def get_loss_accuracy(self, Xdata, Ydata):
        """
        Get the output of the network for some given x as input
        """
        
        fetch_dict = {
            "loss": self.loss,
            "accuracy": self.accuracy,
        }        
        feed_dict = {
            self.x: Xdata,
            self.y: Ydata
        }
        
        ### Propagate train and test data !!!! #####
        fetched_data = self.session.run(fetch_dict, feed_dict)
        
        return fetched_data["loss"], fetched_data["accuracy"]
    
    def run_epoch(self, Xdata, Ydata):
        """ 
        Run an epoch of the training algorithm by means of giving some data
        and fetching the train variable
        """
        
        fetch_dict = {
            "train": self.train,
        }        
        feed_dict = {
            self.x: Xdata,
            self.y: Ydata
        }
        
        ### Propagate train and test data !!!! #####
        fetched_data = self.session.run(fetch_dict, feed_dict)
        
            
        return fetched_data["train"]
        
    def restore_variables(self):
        """
        Restore the variables from the Saved ones
        """
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.variables_path))
        if ckpt:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)


    def close_session(self):
        self.session.close()