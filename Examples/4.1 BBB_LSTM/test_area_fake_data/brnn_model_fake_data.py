from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import util
import reader
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, MultiRNNCell
from tensorflow.contrib.distributions import Normal

from tensorflow.python.client import device_lib

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]

#print(len(gpus))

if len(gpus) == 0:
    global_num_gpus = 1
else:
    global_num_gpus = len(gpus)

"""
    Global variables
"""

def data_type():
    return tf.float32


def sample_posterior(shape, name, prior, is_training):

    """
        Get a sample from the multivariate posterior
    """
    
    
    rho_max_init = math.log(math.exp(prior.sigma_mix / 2.0) - 1.0)
    rho_min_init = math.log(math.exp(prior.sigma_mix / 4.0) - 1.0)
    
    init = tf.random_uniform_initializer(rho_min_init, rho_max_init)
    
    with tf.variable_scope("BBB", reuse = not is_training):
        mu = tf.get_variable(name + "_mean", shape = shape, dtype=data_type())
    
    with tf.variable_scope("BBB", reuse = not is_training):
        rho = tf.get_variable(name + "_rho", shape = shape, dtype=data_type(), initializer=init)
        
    if is_training:
        epsilon = Normal(0.0, 1.0).sample(shape)
        sigma = tf.nn.softplus(rho) + 1e-5
        output = mu + sigma * epsilon
    else:
        output = mu

    if not is_training:
        return output
    
    tf.summary.histogram(name + '_rho_hist', rho)
    tf.summary.histogram(name + '_mu_hist', mu)
    tf.summary.histogram(name + '_sigma_hist', sigma)

    sample = output
    kl = get_kl_divergence(shape, tf.reshape(mu, [-1]), tf.reshape(sigma, [-1]), prior, sample)
    tf.add_to_collection('KL_layers', kl)

    return output


def get_kl_divergence(shape, mu, sigma, prior, sample):
    

    """
    Compute KL divergence between posterior and prior.
    log(q(theta)) - log(p(theta)) where
    p(theta) = pi*N(0,sigma1) + (1-pi)*N(0,sigma2)
    
    shape = shape of the sample we want to compute the KL of
    mu = the mu variable used when sampling
    sigma= the sigma variable used when sampling
    prior = the prior object with parameters
    sample = the sample from the posterior
    """
    
    #Flatten to a vector
    sample = tf.reshape(sample, [-1])
    
    #Get the log probability distribution of your sampled variable
    #So essentially get: q( theta | mu, sigma )
    posterior = Normal(mu, sigma)
    
    
    prior_1 = Normal(0.0, prior.sigma1)
    prior_2 = Normal(0.0, prior.sigma2)
    
    #get: sum( log[ q( theta | mu, sigma ) ] )
    q_theta = tf.reduce_sum(posterior.log_prob(sample))
    
    #get: sum( log[ p( theta ) ] ) for mixture prior
    mix1 = tf.reduce_sum(prior_1.log_prob(sample)) + tf.log(prior.pi_mix)
    mix2 = tf.reduce_sum(prior_2.log_prob(sample)) + tf.log(1.0 - prior.pi_mix)
    
    #Compute KL distance
    KL = q_theta - tf.reduce_logsumexp([mix1,mix2])
    
    return KL


class Prior(object):

    """
        For creating our fixed prior containing the desired 
        properties we want to use in the model.
        
        Setting pi = 1 will lead to a non mixture gaussian
        with mean zero and log var = log_sigma1
    """

    def __init__(self, pi, log_sigma1, log_sigma2):
        self.pi_mix = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = tf.exp(log_sigma1)
        self.sigma2 = tf.exp(log_sigma2)
        sigma_one, sigma_two = math.exp(log_sigma1), math.exp(log_sigma2)
        self.sigma_mix = np.sqrt(pi * np.square(sigma_one) + (1.0 - pi) * np.square(sigma_two))



class BayesianLSTMCell(BasicLSTMCell):

    def __init__(self, X_dim, num_units, prior, is_training, name = None, **kwargs):

        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.n = name
        self.is_training = is_training
        self.num_units = num_units
        self.X_dim = X_dim

    
    #Class call function
    def __call__(self, inputs_i, state):

        with tf.variable_scope("BayesLSTMCell"):
            if self.w is None:

                
                self.w = sample_posterior((self.X_dim  + self.num_units, 4 * self.num_units),
                                              name=self.n + "_weights",
                                              prior=self.prior,
                                              is_training=self.is_training)
    
                self.b = sample_posterior((4 * self.num_units, 1),
                                               name=self.n + "_biases",
                                               prior=self.prior,
                                               is_training=self.is_training)

            # self.w = None; # Should we set it to be None Again so that it is sampled when calling again ?
            
            C_t_prev , h_t_prev = state

            concat_inputs_hidden = tf.concat([inputs_i, h_t_prev], 1)

            gate_inputs =  tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))

            i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

            C_t = (C_t_prev * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i)*self._activation(j))
            h_t  = self._activation(C_t) * tf.sigmoid(o)
            
            State_t = LSTMStateTuple(C_t, h_t)

            return h_t, State_t

class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class BBB_LSTM_Artificial_Data_Input(object):


    def __init__(self, X, Y, batch_size, name=None):
        
        self.batch_size = batch_size  # Size of the batches
        self.num_steps = X[0].shape[0]    # Number of elements of the chain
        
        self.num_chains = len(X)
        self.epoch_size = self.num_chains // batch_size

        self.input_data, self.targets = reader.Artificial_data_producer(X,Y, self.batch_size, name=name)



class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        
        self._input_data = input_.input_data
        size = config.X_dim
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        
        self._targets = input_.targets
        # Construct prior
        prior = Prior(config.prior_pi, config.log_sigma1, config.log_sigma2)
        
        # Fetch embeddings
        inputs = input_.input_data
        # Build the BBB LSTM cells
        cells = []
        for i in range(config.num_layers):
            if (i == 0):
                LSTM_input_size = config.X_dim
            else:
                LSTM_input_size = config.hidden_size
                
            cells.append(BayesianLSTMCell(LSTM_input_size, config.hidden_size, prior, is_training,
                                      forget_bias=0.0,
                                      name="bbb_lstm_{}".format(i)))

        cell = MultiRNNCell(cells, state_is_tuple=True)
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        
        # Forward pass for the truncated mini-batch
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

        # Softmax weights
        softmax_w = sample_posterior((hidden_size, vocab_size), "softmax_w", prior, is_training)
        softmax_b = sample_posterior((vocab_size, 1), "softmax_b", prior, is_training)
        
        logits = tf.nn.xw_plus_b(output, softmax_w, tf.squeeze(softmax_b))
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        
        self._output =  tf.nn.softmax(logits)
        
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
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
        #B = number of batches aka the epoch size
        #C = number of truncated sequences in a batch aka batch_size variable
        B = self._input.epoch_size
        C = self.batch_size
        
        kl_loss = tf.add_n(tf.get_collection("KL_layers"), "kl_divergence")
        
        kl_factor = 1.0/(B*C)
        self._kl_loss = kl_factor * kl_loss
        
        self._total_loss = self._cost + self._kl_loss

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._total_loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost,
               util.with_prefix(self._name, "kl_div"): self._kl_loss,
               util.with_prefix(self._name, "input_data"): self._input_data,
               util.with_prefix(self._name, "output"): self._output,
               util.with_prefix(self._name, "targets"): self._targets,
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
        self._input_data = tf.get_collection_ref(util.with_prefix(self._name, "input_data"))[0]
        self._output = tf.get_collection_ref(util.with_prefix(self._name, "output"))[0]
        self._targets = tf.get_collection_ref(util.with_prefix(self._name, "targets"))[0]
        
        num_replicas = num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def input_data(self):
        return self._input_data
    
    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    @property
    def kl_loss(self):
        return self._kl_loss if self._is_training else tf.constant(0.)

    @property
    def output(self):
        return self._output
    
    @property
    def targets(self):
        return self._targets
    
def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        fetches["kl_divergence"] = model.kl_loss

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and (step % (model.input.epoch_size // 10) == 10 or step == 0):
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * global_num_gpus /
                       (time.time() - start_time)))

            if model._is_training:
                print("KL is {}".format(vals["kl_divergence"]))

    return np.exp(costs / iters)

def fetch_output(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    costs = 0.0
    state = session.run(model.initial_state)
    
    inputs = []
    outputs = []
    fetches = {
        "final_state": model.final_state,
        "output": model.output,
        "input": model.input
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        fetches["kl_divergence"] = model.kl_loss

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h


        vals = session.run(fetches, feed_dict)
        state = vals["final_state"]
        output = vals["output"]
        input_i = vals["input"
                       ]
        outputs.append(output)
        inputs.append(input_i)
        
    return inputs, outputs


#def change_random_seed(seed):
#    global prng
#    prng = np.random.RandomState(seed)
#    tf.set_random_seed(seed)
