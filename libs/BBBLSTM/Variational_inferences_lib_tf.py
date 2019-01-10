from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf

from tensorflow.contrib.distributions import Normal

def data_type():
    return tf.float32


def inv_soft_plus(x):
    return math.log(math.exp(x) - 1.0) 


def get_KL_divergence_Sample(shape, mu, sigma, prior, Z):
    
    """
    Compute KL divergence between posterior and prior.
    Instead of computing the real KL distance between the Prior and Variatiational
    posterior of the weights, we will jsut sample its value of the specific values
    of the sampled weights  W. 
    
    In this case:
        - Posterior: Multivariate Independent Gaussian.
        - Prior: Mixture model
    
    The sample of the posterior is:
        KL_sample = log(q(W|theta)) - log(p(W|theta_0)) where
         p(theta) = pi*N(0,sigma1) + (1-pi)*N(0,sigma2)
    
    Input:
        - mus,sigmas: 
        - Z: Samples weights values, the hidden variables !
    shape = shape of the sample we want to compute the KL of
    mu = the mu variable used when sampling
    sigma= the sigma variable used when sampling
    prior = the prior object with parameters
    sample = the sample from the posterior
    
    """
    
    # Flatten the hidden variables (weights)
    Z = tf.reshape(Z, [-1])
    
    #Get the log probability distribution of your sampled variable
    
    # Distribution of the Variational Posterior
    VB_distribution = Normal(mu, sigma)
    # Distribution of the Gaussian Components of the prior
    prior_1_distribution = Normal(0.0, prior.sigma1)
    prior_2_distribution = Normal(0.0, prior.sigma2)
    
    # Now we compute the log likelihood of those Hidden variables for their
    # prior and posterior.
    
    #get: sum( log[ q( theta | mu, sigma ) ] )
    q_ll = tf.reduce_sum(VB_distribution.log_prob(Z))
    
    #get: sum( log[ p( theta ) ] ) for mixture prior
    mix1 = tf.reduce_sum(prior_1_distribution.log_prob(Z)) + tf.log(prior.pi_mix)
    mix2 = tf.reduce_sum(prior_2_distribution.log_prob(Z)) + tf.log(1.0 - prior.pi_mix)
    p_ll = tf.reduce_logsumexp([mix1,mix2])
    
    #Compute the sample of the KL distance as the substaction ob both
    KL = q_ll -  p_ll
    
    return KL


class Prior(object):

    """
        Class in order to store the parameters fo the prior.
        When initialized it just stores the values to be used later.
        Input:
            - 
        
    """

    def __init__(self, pi, log_sigma1, log_sigma2):
        self.pi_mix = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = tf.exp(log_sigma1)
        self.sigma2 = tf.exp(log_sigma2)
        sigma_one, sigma_two = math.exp(log_sigma1), math.exp(log_sigma2)
        self.sigma_mix = np.sqrt(pi * np.square(sigma_one) + (1.0 - pi) * np.square(sigma_two))
        



def sample_posterior(shape, name, prior, is_training):

    """
        In order to initialize the weights of our network we need to sample from 
        the Variational Posterior. 
        
        This function will be called for any element of the Model that used Bayesian weights.
        
        Since we are using TF, we have to adapt the way we treat the variables.
        In this case we want to optimize the parameters of the Variational Inference. 
        There parameters will be optimized automatically by TF. 
        
        In order to create and retrieve such wanderful variables we use the same function:
        tf.get_variable() to which we can indicate the variable to retrieve indicating:
          - Its name: In this case provided externally to avoid collisions
          - Its shape: We could obtain a subset of the original variable ?
          - Its type: The datatype of the variable
          - An initializer: The first time we use tf.get_variable() if the variable does
            not exist it will be created. If nothing specified it will be created with all 0s
            with a shape given by the parameter shape. We can also specify an initilizer that will
            initialize the variable the first time.
        
        The first time we want to sample from the posterior during training, the variable
        will not exist and it will be sampled from the Prior. The next times it will just be obtained.
        
        In this case the variables are the parameters of the posterior :), the mus and stds.
            
    """
    
    ## Initializer for the weights with the prior for the first sampling
    rho_max_init = inv_soft_plus(prior.sigma_mix / 2.0)
    rho_min_init = inv_soft_plus(prior.sigma_mix / 4.0)
    init = tf.random_uniform_initializer(rho_min_init, rho_max_init, dtype=data_type())
    
    ## Retrieve (and create in the first execution) the mus and rhos
    with tf.variable_scope("BBB", reuse = not is_training):
        mus = tf.get_variable(name + "_mean", shape = shape, dtype=data_type(),initializer=init)
    
#    print (shape)
#    print (is_training)
#    print (mus.dtype)
#    print (mus.shape)
#    print (data_type())
    
    with tf.variable_scope("BBB", reuse = not is_training):
        rhos = tf.get_variable(name + "_rho", shape = shape, dtype=data_type())
    
    # Sample a set of weights from the posterior.
    # Or if we are not training, sample the MAP weights which are the mean.
    
    if is_training:
        epsilon = Normal(0.0, 1.0).sample(shape)
        sigmas = tf.nn.softplus(rhos) + 1e-10
        Samples = mus + sigmas * epsilon
    else:
        Samples = mus

    ## Gather information for tensorflow if we are sampling while training 
    if (is_training):
        # Save the values of the rhos, mus and sigmas of the posteior
        tf.summary.histogram(name + '_rho_hist', rhos)
        tf.summary.histogram(name + '_mu_hist', mus)
        tf.summary.histogram(name + '_sigma_hist', sigmas)
        
        # Save the KL divergence of these sampled weights
        kl = get_KL_divergence_Sample(shape, tf.reshape(mus, [-1]), tf.reshape(sigmas, [-1]), prior, Samples)
        tf.add_to_collection('KL_layers', kl)
        
        
    """
    TODO: So this function will only be called specifically once when the architecture is set 
    and then we do not execute it again ? Tensorflow will only execute the parts that are used when we run 
    a session in which we feed and fetch data ?
    At every session run, only the connecting part will be executed, so for the LSTM what we have to do is:
        - The very first time it is executed, it self.w = null then execute the normal initializer, the rest of
        times just sample the posterior, but just once, then do nothing. For each of the Batches.
    """
    print ("Posterior Sampling: %s"% name)
    return Samples

############################################################################
############ Alternative Versions of Sampling and KL ###########################
############################################################################

#def Independent_MultiGaussian_Sampling(name, mus, stds, shape):
#    """ 
#    This function will draw samples of the given Independent Multivariate Distribution
#    Using TensorFlow with a name so we can track them later.
#    The parameters are:
#            - name: Name of this operation for tensorboard
#            - mus: Vector with the means of the Independent Multivatiate Gaussian.
#            - stds: Vector with the standard deviation of the independent components
#    Output:
#        - 
#    the given multivatiate distribution
#    
#    """
#    with tf.variable_scope("Independent_MultiGaussian_Sampling"):
#    
#        ## TODO: I have no idea why we multiply by 1
#        rhos = tf.multiply(inv_soft_plus(stds), tf.ones(shape))
#        mus = tf.multiply(mus,tf.ones(shape))
#        
#        ## Create TensorFlow variables for the mean and the rhos
#        mus = tf.get_variable(name + "_mean", initializer = mus, dtype=tf.float32)
#        rhos = tf.get_variable(name + "_rho", initializer=rhos, dtype=tf.float32)
#        
#        # Revert back to std for sampling I guess
#        stds = tf.nn.softplus(rhos)
#    
#        # Sample as many parameters as shape says.
#        epsilon = tf.random_normal(mean=0.0, stddev=1.0, name="epsilon", shape=shape, dtype=tf.float32)
#      
#        #random_var = mean + standard_deviation*epsilon
#        samples = tf.add(mus, tf.multiply(stds,epsilon))
#    
#        return samples, mus, stds

