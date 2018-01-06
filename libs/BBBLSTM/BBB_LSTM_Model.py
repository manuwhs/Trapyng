from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import util_autoparallel as util
import reader

from tensorflow.contrib.rnn import MultiRNNCell

import Variational_inferences_lib as VI
import BayesianLSTMCell as BLC
# TODO: Make a study of when the prediction of the network is the better possible (which time-step).
# Probably at the beggining, it should not have any idea about it, predicting the trend maybe ? 
# And then, the more samples it has, the better the prediction of the future is. 
class BBB_LSTM_PTB_Input(object):
    """ 
    This class gathers the data for the model.
    By calling the self.input_data and self.targets it will retrieve
    one of the batches of the data.
    
    THESE OBJECTS ARE CREATED AS OPERATIONS IN THE GRAPH !!
    THEY CANNOT BE CREATED OUTSIDE !
    """

    def __init__(self, batch_size,num_steps, data, name=None):
        
        self.batch_size = batch_size  # Size of the batches
        self.num_steps = num_steps    # Number of elements of the chain
        
        # Num steps should be given by the input chains and could be different for every chain
        # In the initial case I think what happened is that they use this size to chop the file.
        # TODO: The way is implemented is this the only way to do it, the sum_steps has to be the same for all chaines ?
        
        # Number of epochs ! The number of times we run the entire data ?
        # Why is it constrained by the number of steps ?
        
        self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        
        # This reader formats the data in the desired way somehow.
        """
        IMPORTANT:
            In this architecture we do not need to feed data to the model.
            What will happen is, when we fetch the output, it will depend on this
            self.input_data object, calling the reader.ptb_producer function.
            Which will output a different batch anytime we call it, since it references them with
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            
        """
        self.input_data, self.targets = reader.ptb_producer(data, self.batch_size, self.num_steps, name=name)


class BBB_LSTM_Artificial_Data_Input(object):
    """ 
    This class gathers the data for the model.
    By calling the self.input_data and self.targets it will retrieve
    one of the batches of the data.
    
    In this example X and Y are already all the data and targets.
    They are normal lists with dimensions 
        - X[num_chains][num_steps, X_dim]
        - Y[num_chains][num_steps, 1] 
    
    By specifying batch_size, we will gather them in batches and return them
    every time we call the data 
    
    """

    def __init__(self, X, Y, batch_size, name=None):
        
        self.batch_size = batch_size  # Size of the batches
        self.num_steps = X[0].shape[0]    # Number of elements of the chain
        
        self.num_chains = len(X)
        # Num steps should be given by the input chains and could be different for every chain
        # In the initial case I think what happened is that they use this size to chop the file.
        # TODO: The way is implemented is this the only way to do it, the sum_steps has to be the same for all chaines ?
        
        # Number of epochs ! The number of times we run the entire data ?
        # Why is it constrained by the number of steps ?
        
        self.epoch_size = self.num_chains // batch_size
        
        # This reader formats the data in the desired way somehow.
        """
        IMPORTANT:
            In this architecture we do not need to feed data to the model.
            What will happen is, when we fetch the output, it will depend on this
            self.input_data object, calling the reader.ptb_producer function.
            Which will output a different batch anytime we call it, since it references them with
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            
        """
        self.input_data, self.targets = reader.Artificial_data_producer(X,Y, self.batch_size, name=name)



class BBB_LSTM_Model (object):
    
    def __init__(self, is_training, config, input_):
        """
        This initializer function will read the hyperparameters, from that it will 
        set the atchitecture of the network.
        
        The is_training flag is nice to build the network. If it is not for training then we do not
        need to builf to the graph the loss function and optimizer.
        
        
        """
        # Variable to know if the model is being used for training 
        self._is_training = is_training
        # TODO: This is the structure we just saw...
        self._input = input_
        
        # Setting the chains properties
        self.batch_size = config.batch_size
        self.num_steps = input_.num_steps
        
        input_data_ids = input_.input_data
        targets = input_.targets
        
        # Setting the architectute properties
        # Dimensionality of the input !! 
        # TODO: For now we set it the same as the hidden_size. Probably for matrix concatenation purposes ?
       
        # Dimensionality of the output ! In the case of classification, the cardinality of the output
        Y_cardinality = config.Y_cardinality # Size of the output
        
        # Construct prior
        prior = VI.Prior(config.prior_pi, config.log_sigma1, config.log_sigma2)
        
        ########################################################################
        #############  Transform Categorial values (words) into real values vectors ############
        ########################################################################
        # Fetch embeddings
#        with tf.device("/cpu:0"):
#            embedding = VI.sample_posterior([vocab_size, size], "embedding", prior, is_training)
#            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        # If we have discrete input X and we want to embed them in random vectors of size "size"
        # We also need to include the cardinality of the output Y.
#        if (type(config.X_dim) != type(None)):
        if (config.embedding == True):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        "embedding", [Y_cardinality, config.X_dim], dtype=VI.data_type())
                inputs = tf.nn.embedding_lookup(embedding, input_data_ids)
            
            X_dim = config.X_dim
        else:
            X_dim = config.X_dim# inputs.get_shape()[-1].value
            
#            inputs = tf.get_variable("Continous_data_input", [self.batch_size,self.num_steps, X_dim], dtype=VI.data_type(), trainable = False) 
#            inputs.assign(input_data_ids)
#            
#            caca = tf.zeros_initializer(tf.int32)((self.batch_size,Y_cardinality, tf.int32))
#            targets = tf.get_variable("Discrete_Target", [self.batch_size,Y_cardinality], dtype=tf.int32, trainable = False, 
#                                      initializer = caca) 
#            targets.assign(input_.targets)
            
        
#            inputs = tf.Variable(input_data_ids, trainable = False)
#            targets = tf.Variable(input_.targets, trainable = False)
            
            inputs = input_data_ids
            targets = input_.targets
            
        # These are the chains in the Batch. They are represented by a 3D tensor with dimensions
        #     - size_epoch: Number of chains in the batch
        #     - num_steps: Number of elements of the chain
        #     - D:   Dimensionality of the elements of the chain. 
        # TODO: maybe due to the initial embedding that has to be done, all inputs are given when defining the model,
        #       we do not want that, we want them to be in a way where do the preprocessing before and we have chains as placeholder.

        input_chains = inputs[:, :, :]
        
        print ("-----------------------------")
        print ("Input Batch X shape", inputs.shape)
        print ("Input Batch Y shape", targets.shape)
        print ("Input_size: %i"%X_dim)
        print ("Output_size: %i"%Y_cardinality)
        print ("Number of chains in a batch: %i"%self.batch_size)
        print ("Number of elements in a chain: %i"%self.num_steps)
        print ("Number of hidden state neurons LTSM: %i"%config.hidden_size)
        
        ########################################################################
        ############# Start Building the Architecute of the Network ############
        ########################################################################
        
        ######################################################################
        ################  Build and Stack BBB LSTM cells ################
        cells = []
        for i in range(config.num_layers):
            if (i == 0):
                LSTM_input_size = X_dim
            else:
                LSTM_input_size = config.hidden_size
                
            cells.append(BLC.BayesianLSTMCell(LSTM_input_size, config.hidden_size, prior, is_training,
                                      forget_bias=0.0,
                                      name="bbb_lstm_{}".format(i)))
        # The following line will stack the LSTM cells together
        # They just need to follow the interface that we already wrote
        # Notice we use  state_is_tuple=True since the LSTM cells have 2 states C_t and h_t
        DeepLSTMRNN = MultiRNNCell(cells, state_is_tuple=True)
        
        # Initialize the state values to 0 ? 
        # TODO: We need to provide info about the Batch size ? That is the number of chains
        # we want to compute the output at once. 
            
        #####################################################################################
        ################  Propagate the chains in the batch from input to output ################
        
        # Initialization.
        # This is the initial state for the LSTM when we feed it a new chain (is it just the 0s) probably. Then it should output the conditional most lilkely word.
        # We need to give it the batch_size because we are going to propagate the chains in parallel. 
        # initial state will have dimensions [batch_size, (LSTM_hidden_size, LSTM_hidden_size)] since each state of the LSTM is made of the previous 
        self._initial_state = DeepLSTMRNN.zero_state(config.batch_size, VI.data_type())
        state = self._initial_state


        # Forward pass for the truncated mini-batch
        # hs_o: This list will contain in each of its elements, 
        #         the hidden state of the last LSTM of the network
        #         for each of the number of steps (length of the chains that is has to be the same for every chain).
        # Each of this hidden states has dimensions [LSTM_hidden_size, num_batch] since we are computing in parallel for all chains in the batch.

        # Now we propagate the chains in parallel and the initial state through the Deep Bayesian LSTM.
        # At each time step we will save the hidden state of the last LSTM to convert it later to the real output and being able
        # to compute the cost function and the output !

        # TODO: This is probably why we want the chains to have the same length. Also maybe to not having to worry later to weight the
        # cost functions by the length of the chains. Anyway... for now we will just accept it.

        hs_o = []                       
        with tf.variable_scope("RNN"):        # We put all the LSTMs under the name RNN.
            for time_step in range(self.num_steps):  # For each element in the chain
                if (time_step > 0):   # Maybe this is so that we do not create the LSTMS a lot of times in the TensorBoard ?
                    tf.get_variable_scope().reuse_variables()
                
                # Now we start feeding the time_step-th element of each of the chains at the same time to the network, obtaining the state for

                (cell_output, state) = DeepLSTMRNN(input_chains[:,time_step,:], state)
                hs_o.append(cell_output)
        print (["size output state LSTM", cell_output.shape])
        
#        print ("Num steps: %i"%self.num_steps)
        
        # Now we concatenate all the hidden spaces of dimension  [num_batch, LSTM_hidden_size] 
        # into in the list with dimension [num_batch x step_size, LSTM_hidden_size]. At the end of the day
        # all of the hidden spaces will be multiplied by the same weights of the dense softmax layer so we concatenate all of the
        # output hidden spaces for later multiplication.
        hs_o = tf.reshape( tf.concat(hs_o, 1), [-1, config.hidden_size])
        
        print (["Size of the Concatenated output state of the last LSTM for all chains in batch and time-steps in a batch", hs_o.shape])
        ######################################################################
        ################  Build the output layer ############################

        # In our case the output later is just a dense layer that transforms the hidden space
        # of the last LSTM into the prediction of each discrete output (word), applying a softmax 
        # function to the output of the neurons.
        # The parameters of this layer are just the Weights and biases of it.
        
        # The next call function will create the weights if they have not been create before.
        # Identified by the names ""
        # TODO: Not really a TODO, but the important part here is that we changed size vy config.hidden_size
        softmax_w = VI.sample_posterior((config.hidden_size  , Y_cardinality), "softmax_w", prior, is_training)
        softmax_b = VI.sample_posterior((Y_cardinality, 1), "softmax_b", prior, is_training)
        
        print ("Shape of the weights of the output Dense layer",softmax_w.shape)
        print ("Shape of the weights of the output Dense layer",softmax_b.shape)
        ## We propagate the hidden spaces through the network in order to obtain the outout of the network before
        ## the softmax function, which is called the logits. This logits will have dimensions 
        ## [num_batch x step_size, LSTM_hidden_size] that we need to break down further.

        # Logits are the input to the softmax layer !
        logits = tf.nn.xw_plus_b(hs_o, softmax_w, tf.squeeze(softmax_b))
        # We reshape it back to the proper form [chain, sample, output]
        
        print ("Shape of logits after multiplication of ohs", logits.shape)
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, Y_cardinality])
        print ("Shape of logits after reshpaing", logits.shape)
        
        # We can compute the output of the chains !
        # TODO: maybe do not execute this line in the training model to save computation ? Maybe it wouldnt be executed anyway ?
        self._output =  tf.nn.softmax(logits)

        """ This is finally the output of the batch, our prediction of the word,
            for each of the words in the batch. Since we have:
                - self.batch_size number of chains in the batch
                - Each chain has the same number of words: self.num_steps
                - The prediction of each word is the probability of each of the vocab_size variables
        """
        
        #####################################################################################
        ################  Setting the Loss function  ################
        #####################################################################################

        #B = number of batches aka the epoch size
        #C = number of truncated sequences in a batch aka batch_size variable
        B = self._input.epoch_size
        C = self.batch_size
        
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([self.batch_size, self.num_steps], dtype=VI.data_type()),
            average_across_timesteps=False,
            average_across_batch=False)

        # Update the cost
        # Remember to divide by batch size
        self._cost = tf.reduce_sum(loss) / self.batch_size
        self._kl_loss = 0.
        self._final_state = state
        
        if not is_training:
            return

        #Compute KL divergence

        ## We get the KL loss that was computed during the sampling of the variational posterior !!
        
        kl_loss = tf.add_n(tf.get_collection("KL_layers"), "kl_divergence")
        
        self._kl_loss =  kl_loss /(B*C)
        
        # Compute the final loss, this is a proportion between the likelihood of the data (_cost)
        # And the KL divergence of the posterior 
        
        # TODO: Remove increased by 2 the cost so that the total cost is more influenced
        # on  the data !
        self._total_loss = self._cost + self._kl_loss
        
        #####################################################################################
        ################  Setting the training algorithm  ################
        #####################################################################################
        
        ## Set the trainable variables, the variables for which the gradient with respect to the loss function
        # will be computed and will be modified by the optimizer when the session is run :) 
            
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._total_loss, tvars),
                                          config.max_grad_norm)
        
        
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(VI.data_type(), shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        #####################################################################################
        ################  Methods of the model !! ####################################
        #####################################################################################
    
    # Set the new learining rate of the model.
    # This is done by doing a session run where we fee the new learning rate and fetch a variable equal to TensorFlow assignation that
    # will change the value of the model when called ? It that how it works ?
    # TODO 
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
 
    """
    TODO: 
    Functions to load the operations ? 
    For this initial thing that we first define the models and then destroy de graph.
    Do we need them for later anyway ?

    TODO: In the fake generation, then I computed the P(up) given the current and prev signal, but not given the previos ones ? So could we do better than that ? 
    Or  worse since this is looking at the future ? dunno. Can we actually learn the probability of it going up or down given the previous signal ? Actually we can get the ideal I guess.
    Given that we know the true signal, then we only have to compute the conditional probability given the previous noise... Worth a try.
    """

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost,
               util.with_prefix(self._name, "kl_div"): self._kl_loss,
               util.with_prefix(self._name, "output"): self._output
               }
        
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            
        for name, op in ops.items():
            tf.add_to_collection(name, op)
            
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self, num_gpus = 1):
        """Imports ops from collections."""
        
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        self._kl_loss = tf.get_collection_ref(util.with_prefix(self._name, "kl_div"))[0]
        self._output = tf.get_collection_ref(util.with_prefix(self._name, "output"))[0]
        
        num_replicas = num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    """
    Here we set the variables that we want to access later for feeding and fetching
    when calling a session function. It is an interface to the variables of the session.
    """

    ############ Variables related to the State of RNNs ####################
    
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def final_state(self):
        return self._final_state

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    ####################  Loss and cost functions ####################
    @property
    def cost(self):
        return self._cost

    @property
    def total_loss(self):
        return self._total_loss

    @property
    # Since for the models that were created not for training, we do not run the lines that create 
    # the self._kl_loss variable. If asked for them, then we return 0 instead in a tensor.
    def kl_loss(self):
        return self._kl_loss if self._is_training else tf.constant(0.)


    #################### Hyperparameters !! ####################
    @property
    def lr(self):
        return self._lr
    
    #################### Data ##############################
    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output
    
    #################### Operations ##############################

    @property
    def train_op(self):
        return self._train_op
    
    def fit_batch():
        """
        This function aims to train the model parameters with the batch of data provided
        """
        pass
   
    def predict():
        """
        This function aims to return the prediction of a batch of chains given to the model.
        """
        pass
    def get_loss():
        """
        If we give it the input data and the target, it wil compute the loss and those things.
        """
        pass


def run_epoch(session, model, eval_op=None, verbose=False, num_gpus = 1):
    """ This function trains the model for a given epoch. 
        This is done in the following way:
        For each of the epochs:
            - It fetches the final cost (LogLikelihood of the data in the batch)
              and the final LSTM state of the first layer. This is done because fetching the cost
              we force the propagation of the data all the way to the cost and we fetch the
              state because the next batch, instead of having the initial LSTM state = 0, it has 
              the ending one of the previos Batch. 
              TODO: IMPORTANT: This is good since we choped contiguous sentences from a text
              so some depende between the last sample of the one batch and the first sample of the next
              is expected. But in general is this contnuity does not exist, it should be set to 0.
            - TODO: For now it does not really feed anything to the network, it was already prefed,
                    this should be changed.
        
        For
        """
    ## Variables to show how much time it is taking and the Accumulated Perplexity
    start_time = time.time()
    costs = 0.0
    iters = 0

    # TODO: Each batch is going to reuse the state of the previous batch ? 

    state = session.run(model.initial_state)
    
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        fetches["kl_divergence"] = model.kl_loss

    """
    Each epoch is made of a number "epoch_size" of batches.
    For each batch we run the Model
    """

    for step in range(model.input.epoch_size):
        
        # We feed the previous LSTM states at the end of the batch
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
            
        ##########################################
        # Run the session by fetching the output of our network and giving the input.
        vals = session.run(fetches, feed_dict)
        
        # Obtain the variables fetched
        cost = vals["cost"]
        state = vals["final_state"]
#        output = vals["output"]
        
#        print (output[3,10,0:10])
#        print (np.sum(output[3,10,:]))
        

        ################# Compute some verbose at every set of of Batches #################
        """
        Notice we print the accumulated perplexity of the training data. 
        This is a bit missleading since the model is changing at every Batch.
        To get the real perplexity of the training data, we should rerun the 
        training data without training.
        """
        costs += cost
        iters += model.input.num_steps
        if (verbose and ((step % 100) == 0 or step == 0)):
            print ("---------------- Batch %i/%i ----------------" %(step+1, model.input.epoch_size))
            print("%.3f pct completed.  Perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * num_gpus /
                       (time.time() - start_time)))

            if model._is_training:
                print("KL is {}".format(vals["kl_divergence"]))

    return np.exp(costs / iters)