from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, MultiRNNCell
from tensorflow.contrib.distributions import Normal
from tensorflow.python.client import device_lib

import Variational_inferences_lib as VI

class BayesianLSTMCell(BasicLSTMCell):

    def __init__(self, X_dim, num_units, prior, is_training, name = None, **kwargs):
        """
        In the initialization of the Cell, we will set:
                - The internal weights structures w and b. Which will hold the weights and biases
                  for the 4 gates of the LSTM cell. 
                - The Prior of the weights: Needed to compute the first set of weights before seeing any data
                  As wwll as the number of units in each of the 4 gates
        and the rest of the parameters that the original BasicLSTMCell could need.
        
        NOTICE: THE FIRST LSTM IT HAS AS INPUT THE DIMENSION OF X.
                THE REST ARE FED WITH THE OUTPUT STATE OF THE PREVIOUS LSTM, 
                   WHICH HAS DIMENSION OF THE HIDDEN SPACE
        
        """
        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.n = name
        self.is_training = is_training
        self.num_units = num_units
        self.X_dim = X_dim
    
    """
        A note on the shape for the sampling of weights and biases:
        
        The number of weights we want to sample is determined by the hidden state
        from the previous cell, the input data, and of course the number of gates
        in a single cell.
        
        Assume that num_units = 10:
            
        An LSTM cell contains 4 gates. Each gate will be composed by a Neural Network
        with "num_units" output neurons that will represent the hidden space of the cell.
        
        If we do not have any hidden units in the gates, then the input and the output are connected
        And we will have 4 x num_units neurons. 
        The input space of each of the gates is [X , S_{t-1}] so we have an input dimension of [ D + num_units]
        since the size of the previous state is the same as the size of the current.
    """
    
    
    #Class call function
    def __call__(self, inputs, state):
        """
        Compute the hidden state h_t (output of the network) and the cell state C_t 
        from a given input and the previous state [C_t, h_t]
        
        """
        with tf.variable_scope("BayesLSTMCell"):
            if self.w is None:

#                size = inputs.get_shape()[-1].value
                
                print (["------- Size input LSTM: ", inputs.shape])
                print (["------- Dim input specified ", self.X_dim])
#               print (["num units LSTM: ", self.num_units])
                
                self.w = VI.sample_posterior((self.X_dim  + self.num_units, 4 * self.num_units),
                                              name=self.n + "_weights",
                                              prior=self.prior,
                                              is_training=self.is_training)
    
                self.b = VI.sample_posterior((4 * self.num_units, 1),
                                               name=self.n + "_biases",
                                               prior=self.prior,
                                               is_training=self.is_training)

            # Get the cell and hidden state from the previous cell [C_t-1, h_t-1]
            C_t_prev , h_t_prev = state
            #Vector concatenation of previous hidden state and embedded inputs
            concat_inputs_hidden = tf.concat([inputs, h_t_prev], 1)
            # Compute the Z = Wx + b for each of the 4 networks at once !
            gate_inputs =  tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))
            
            # Split data up for the 4 gates
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

            # Compute the new cell 
            C_t = (C_t_prev * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i)*self._activation(j))
            h_t  = self._activation(C_t) * tf.sigmoid(o)
            
            #Create tuple of the new state
            State_t = LSTMStateTuple(C_t, h_t)

            return h_t, State_t
        



