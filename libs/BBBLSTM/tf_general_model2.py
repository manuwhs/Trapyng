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

import graph_lib as gl
import numpy as np
## Import standard libraries
import tensorflow as tf
# Import libraries
import utilities_lib as ul
import tf_loss_functions as tflofu
import tf_checkpoints_utils as tfchul
import tf_models_utilities as tfmul 

class General_Model:
    """
    This Model is general purpose, it provides a set of parts of the graph. 
    The different parts would be interconnected with interfaces to better assemply
    """
    def __init__(self, name = "General_model1"):
        """
        Function to 
        """
        
        self.name = name
        self.saver = None;
        

    def _create_placeholders(self, shape_x = [None, 1], shape_y =  [None, 1]):
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
            
        None shape value means variable size.
        
        """
        with tf.name_scope("Input_Data"):
            x = tf.placeholder(tf.float32, shape_x, name='x_pl')
            y = tf.placeholder(tf.float32, shape_y , name='y_pl') # float64

        #### Store the input variables in the model
        self.x = x;
        self.y = y;
        
        return x,y

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        pass
    
    

    def _create_weight_initializer(self,init_config):
        """
        DEFINE INITILIZATION OF WEIGHTS  
        #Define initializer for the weigths
        How the weights are initialized is very important for how well the network 
        trains. We will look into this later, but for now we will just use a normal 
        distribution.
        """
        if (init_config.name == "truncated_normal"):
            weight_initializer = tf.truncated_normal_initializer(stddev=init_config.std)
        self.weight_initializer = weight_initializer

    def _create_IO_graph(self, model_config):
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

        with tf.name_scope("Trainable_Variables"):
            
            with tf.name_scope("Layer_1_variables"):
                W_1 = tf.get_variable('W1', [model_config.input_dim, model_config.num_hidden1], 
                                      initializer= self.weight_initializer,  dtype = tf.float32)
                b_1 = tf.get_variable('b1', [ model_config.num_hidden1],
                                      initializer=tf.constant_initializer(0.0),  dtype = tf.float32)
            with tf.variable_scope('Layer_2_variables'): 
                W_2 = tf.get_variable('W2', [model_config.num_hidden1, model_config.num_hidden2],
                                      initializer=self.weight_initializer)
                b_2 = tf.get_variable('b2', [model_config.num_hidden2], 
                                      initializer=tf.constant_initializer(0.0))
            with tf.variable_scope('Output_layer_variables'): 
                W_o = tf.get_variable('Wo', [model_config.num_hidden2, model_config.output_dim],
                                      initializer=self.weight_initializer)
                b_o = tf.get_variable('bo', [model_config.output_dim], 
                                      initializer=tf.constant_initializer(0.0))
                
        with tf.name_scope("State_variables"):
            s = tf.Variable(initial_value = -10, # Initialization, it knows rank 1
                            trainable = False,     # Can be trained by tf
                            dtype = tf.float32,   # Type of the variable
                            name = "State")    # Name of the Variable
        
        """
        3. OPERATIONS
            Define the set of operation that are made between the tensors in order to
            generate the output of the model. In this case we just multiply input and output.
        
        """
        with tf.name_scope("Network_Operations"):
            # We can use normal operators over tensors or the methods in tensorflow
            
            with tf.variable_scope('Layer1'): 
                layer1_z = tf.matmul(self.x, W_1) + b_1    # Multiplication
                layer1_o = tfmul.apply_nonlinearity(layer1_z,model_config.non_linearity1)               # Nonlinear funciton
                
            with tf.variable_scope('Layer2'): 
                layer2_z = tf.matmul(layer1_o, W_2) + b_2    # Multiplication
                layer2_o = tfmul.apply_nonlinearity(layer2_z,model_config.non_linearity1)    

            with tf.variable_scope('OutputLayer'):
                output_z = tf.matmul(layer2_o, W_o) + b_o    # Multiplication
                output = tfmul.apply_nonlinearity(output_z,model_config.non_linearity1)

    #### Store the output variables in the model
        self.output = output;
        self.variables = [W_1,b_1,W_2,b_2, W_o, b_o, s];
        self.s = s
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
                loss = tflofu.huber_loss(self.y,self.output,loss_config.delta)
                
                
            elif(loss_config.name == "MSE"):
                loss = tf.reduce_sum(tf.square(self.y-self.output), name = "MSE_loss")
                
        with tf.variable_scope('performance'):
            RMSE = tf.sqrt(tf.reduce_sum(tf.square(self.y-self.output), name = "MSE_performance"))
            # Tensorboars. Variable that we want to be saved during training so that we do not need to compute it later.
            accuracy = tf.summary.scalar("performance/accuracy", RMSE)

        self.loss = loss;
        self.accuracy = accuracy;
        
        return loss, accuracy


    def _create_optimizer(self, train_config):
        with tf.name_scope("Optimizer"):
            # First we create the optimizer object.
            
            if (train_config.name == "GradientDescent"):
                if (0):  # Extender version
                    # Defining our optimizer
                    optimizer_GD = tf.train.GradientDescentOptimizer(learning_rate = train_config.lr)
                    # Computing our gradients
                    grads_and_vars_GD = optimizer_GD.compute_gradients(self.loss)
                    # Applying the gradients
                    train = optimizer_GD.apply_gradients(grads_and_vars_GD, global_step = self.global_step)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(train_config.lr)
                    # Then we indicate the variable to minimize.
                    train = optimizer.minimize(self.loss,global_step = self.global_step)
                        
            elif(train_config.name == "Adam"):
                optimizer = tf.train.AdamOptimizer(learning_rate= train_config.learning_rate,
                                                   beta1=train_config.Adam_beta1,beta2=train_config.Adam_beta2,epsilon=train_config.Adam_epsilon,name='Adam' )
                train = optimizer.minimize(self.loss, global_step = self.global_step)
                
            elif(train_config.name == "RMSProp"):
                optimizer = tf.train.RMSPropOptimizer(learning_rate= train_config.learning_rate,
                    decay=train_config.RMSProp_decay, momentum=train_config.RMSProp_momentum, 
                    epsilon=train_config.RMSProp_epsilon,use_locking=train_config.RMSProp_use_locking,  
                    centered=train_config.RMSProp_centered, name='RMSProp')
                train = optimizer.minimize(self.loss,global_step = self.global_step)
        
        
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

    def build_graph(self,model_config, init_config, loss_config, train_config):
        """
        Call all the builders of the graph :)
        """
        self._create_placeholders();
        self._create_embedding()
        self._create_weight_initializer(init_config)
        self._create_IO_graph(model_config);
        self._create_loss(loss_config);
        
        if (train_config.saver):
             self._create_saver(train_config.variables_path);
             
        self._create_optimizer(train_config);
        
        self._create_initializer()
        session = tf.Session();
        self.session = session
        
    def initialize(self ):
        """
        Initialize the variables of the network
        """
    
        # Run the session to initialize all variables of the model.
        self.session.run(self.initializer)
        
        
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
        and fetching the train variable. If the fetch the optimizer, it will train  and update global_step.
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
        
        self.session.run(self.s.assign(self.s + 3))
            
        return fetched_data["train"]
    
    def close_session(self):
        self.session.close()
        
    ######### SAVING AND RESTORING VARIABLES ##############
    def _create_saver(self, path):
        """
        Saver to automatically save:
            - Variables
            - Accuracy / Loss (for later)
            
        It can be power every certain global step or time
        """
        
        tfchul.create_saver(self, path)
        
    def restore_variables(self, checkpoint_name = None):
        """
        Restore the variables from the Saved ones
        """
        return tfchul.restore_variables(self, checkpoint_name)

    def return_variables(self):
        tfchul.return_variables(self);
        
    def save_variables(self):
        tfchul.save_variables(self);
    
    def get_global_step(self):
        return self.session.run(self.global_step)
    
    def remove_previous_checkpoints(self):
        """
        Remove the previous checkpoints in the folder
        """
        ul.remove_files(self.variables_path)
        
        