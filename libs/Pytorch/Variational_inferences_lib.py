import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.nn.parameter import Parameter
import pyTorch_utils as pytut
import math

# TODO: REMOVE THE NEED OF CALLING IT HERE
dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)

def log_norm(x, mu, std):
    """Compute the log pdf of x,
    under a normal distribution with mean mu and standard deviation std."""
    
#    print ("X device: ", x.device)
#    print ("mu device: ", mu.device)
#    print ("std device: ", std.device)
    x = x.view(-1)
    mu = mu.view(-1)
    std = std.view(-1)
    return -0.5 * torch.log(2*np.pi*torch.pow(std,2))  \
            - 0.5 * (1/torch.pow(std,2))* torch.pow( (x-mu),2) 
            
def inv_softplus(x):
    """
    Compute rhos from sigmas
    """
    return torch.log(torch.exp(x) - 1.0) 

def softplus(x):
    """
    compute sigmas from rhos
    """
    return  torch.log(torch.exp(x) + 1.0) 

def logsumexp(x):
    return torch.logsumexp(x, 0)

def get_KL_divergence_Samples(mu, sigma, Z, prior, mu_prior_fluid = 0):
    """
    Compute KL divergence between the posterior sampled weights  their prior.
	We just compute the KL divergence for the specific sampled posterior weights,
	not the whole KL between the Variational Posterior q(w) and Prior p(w) distributions.
    This has proven to be enough.
    
    In this case:
        - Posterior q(w): Multivariate Independent Gaussian.
        - Prior p(w):	Mixture model given by the prior object.
		The prior is the same for all weights and the posterior is different for all weights.

    The sample of the KL is:
        KL_sample = log(q(W|theta)) - log(p(W|theta_0)) where
			p(theta) = pi*N(0,sigma1) + (1-pi)*N(0,sigma2)
    
    Input:
        - Z: Sampled weights values from the posterior, the hidden variables !
		     They have the shape: [out_features, in_features] for weights.
                                  [out_features]               for biases
                                  
		- mu = Same shape as Z, every sample weight has its own mean
		- sigma = Same shape as Z, every sample weight has its own std
		- prior = the prior object with parameters
        - mu_prior_fluid: Allows us to put a different mean prior for the weights and biases
        like the one needed for the Highway layer.
    """
#    print ("-------------")
#    print ("mu_shape", mu.shape)
#    print ("sigma_shape", sigma.shape)
#    print ("Z_shape", Z.shape)
#    
#    # Flatten the hidden variables (weights)
#    Z = Z.reshape([-1])
#    print ("Z_shape_flattened", Z.shape)
    
    # Now we compute the log likelihood of those Hidden variables for their
    # prior and posterior.
    
    # FIRST: get: sum( log[ q( theta | mu, sigma ) ] )
    #    q_ll = torch.sum(log_norm(Z),mu, sigma)
    q_ll = torch.sum(log_norm(Z,mu, sigma))
    
    # SECONG: get: sum( log[ p( theta ) ] ) for mixture prior
    mix1 = torch.sum(log_norm(Z,torch.tensor(mu_prior_fluid, dtype = dtype,device = device),torch.tensor(prior.sigma1, dtype = dtype,device = device))) 
    mix2 =  torch.sum(log_norm(Z,torch.tensor(mu_prior_fluid, dtype = dtype,device = device),torch.tensor(prior.sigma2, dtype = dtype,device = device))) 

    if(1):
#        p_ll = np.max(combination)
#        p_ll = torch.tensor(p_ll, dtype = dtype,device = device)
        p_ll = mix2
    else:
        combination = [mix1+ np.log(prior.pi_mix),
                                       mix2+ np.log(1.0 - prior.pi_mix)]
        combination = torch.tensor(combination, dtype = dtype,device = device)
        p_ll = logsumexp(combination)

#    print (mix1,mix2,p_ll )
#    p_ll = mix1
#    print (p_ll)
    #Compute the sample of the KL distance as the substaction of both
    KL = q_ll -  p_ll
    return KL


class Prior(object):

    """
        Class in order to store the parameters for the prior.
		The prior is shared by all weights in the network
        When initialized it just stores the values to be used later.
        Input:
            - 
    """
    def __init__(self, pi, log_sigma1, log_sigma2):
        """
        log_sigma1: Logarithm of the std of smaller std Gaussian component
        log_sigma2: Logarithm of the std of bigger std Gaussian component
        pi: Probability of the first Gaussian component in the mixture
        """
        self.pi_mix = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = np.exp(log_sigma1)
        self.sigma2 = np.exp(log_sigma2)
        
        # Approximation of the std of the mixture, it could be used for initialization of variances of the
        # posterior.
        self.sigma_mix = np.sqrt(pi * np.square(self.sigma1) + (1.0 - pi) * np.square(self.sigma2))
    
        self.mu_bias = 0
        self.mu_weight = 0
        
    def standarize(self, Ninput):
        """
        We divide the variances by the number of inputs in the neuron.
        Using this approach forces that this prior object shoul not be shared 
        with other layers in the network so
        """
    
        print ("Standarizing Prior: ", "Ninput: ", Ninput, "simga1: ", self.sigma1, "simga2: ", self.sigma2 )
        self.sigma1 = self.sigma1/np.sqrt(Ninput)
        self.sigma2 = self.sigma2/np.sqrt(Ninput)
        self.log_sigma1 = np.log(self.sigma1)
        self.log_sigma2 = np.log(self.sigma2)
    
    def get_standarized_Prior(self, Ninput):
        standarized_prior = Prior(self.pi_mix, self.log_sigma1, self.log_sigma2)
        standarized_prior.standarize(Ninput)
        standarized_prior.mu_bias =self.mu_bias
        standarized_prior.mu_weight  =self.mu_weight
        return standarized_prior
    
def initialization_sample_posterior(size, prior):
    """
    NOT USED ANYMORE. But Maybe it should...
    """
    # The first time, we cannot sample form the posterior, we do it from the prior
    samples_prior = Normal(0.0, prior.sigma_mix).sample(size) #.to(device = device, dtype = dtype)
    return samples_prior

def init_rho(size, prior, type = "Linear"):

    # Initializer for the weights with the prior for the first sampling
    if(type == "Linear"):
        rho_max_init = inv_softplus(torch.tensor(prior.sigma2).to(device = device, dtype = dtype))
        rho_min_init = inv_softplus(torch.tensor(prior.sigma1).to(device = device, dtype = dtype))
        samples_rho= Uniform(rho_min_init, rho_max_init).sample(size).to(device = device, dtype = dtype)
    
#    print (sigma_max_init, sigma_min_init)
#    print (samples_sigma)
    return samples_rho

def init_mu(size, prior, Ninput, type = "Linear"):

    # Initializer for the weights with the prior for the first 
    if(type == "Linear"):
        stdv = 1. / math.sqrt(Ninput)
        samples_mu = Uniform(-stdv, stdv).sample(size).to(device = device, dtype = dtype)
    elif(type == "LinearSimilarity"):
        if(size[0] == 1): # The biasç
            samples_mu = torch.tensor([0.0]).to(device = device, dtype = dtype)
        else:
            std = math.sqrt(6 / (Ninput + 1))
            samples_mu = Uniform(-std, std).sample(size).to(device = device, dtype = dtype)
#    print (sigma_max_init, sigma_min_init)
#    print (samples_sigma)
    return samples_mu

def sample_posterior (mus, sigmas):

    """"
	For every training batch, we need to sample the weights from the Variational Posterior.
    This function will be called for any element of the Model that used Bayesian weights.
        
    The first time we want to sample from the posterior during training, the variable
    will not exist and it will be sampled from the Prior. The next times it will just be obtained.
        
    In this case the variables are the parameters of the posterior :), the mus and stds.
            
    """
    # Reparametrization !!
    # The eps for the reparametrizaiton trick
    eps = Normal(0.0, 1.0).sample(mus.size()).to( dtype = dtype,device = device)
#        sigmas = softplus(rhos)
#    print (sigmas.device)
#    print (mus.device)
#    print (eps.device)
    posterior_samples = eps.mul(sigmas).add(mus)
    return posterior_samples

def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

def remove_by_condition(cond,vec):
    zeros_vec  = torch.zeros_like(vec)
    vec = where(cond, zeros_vec, vec)
#    in_bound_indices = torch.nonzero(vec).squeeze(1)
#    vec = torch.index_select(vec, 0, in_bound_indices)
    return vec

def trim_LinearVB_weights(VBmodel,  mu_sigma_ratio = 2):
    with torch.no_grad():
        sigma_W = softplus(VBmodel.rho_weight)
        mu_W = VBmodel.mu_weight
        
        sigma_b = softplus(VBmodel.rho_bias)
        mu_b = VBmodel.mu_bias
        
        ratio_W = torch.abs(mu_W)/sigma_W
        ratio_b = torch.abs(mu_b)/sigma_b
        
        cond = ratio_W < mu_sigma_ratio
        VBmodel.mu_weight.data = remove_by_condition(cond, mu_W)
        size_w = mu_W.nelement()
        removed_w = float(torch.sum(cond))
        
        ## Biases !
        cond = ratio_b < mu_sigma_ratio
        VBmodel.mu_bias.data = remove_by_condition(cond, mu_b)
        size_b = mu_b.nelement() 
        removed_b = float(torch.sum(cond))
        
        return size_w, removed_w, size_b, removed_b