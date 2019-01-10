# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

import Variational_inferences_lib as Vil
import math
class LinearVB(nn.Module):
    """
    Bayesian Linear Layer with parameters:
        - mu: The mean value of the 
        - rho: The sigma of th OR sigma
    
    """
    def __init__(self, in_features, out_features, bias=True, prior = None):
        super(LinearVB, self).__init__()
        self.type_layer = "linear"
        # device= conf_a.device, dtype= conf_a.dtype,
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.posterior_mean = False # Flag to know if we sample from the posterior mean or we actually sample
        
        ## If no prior is specified we just create it ourselves
        if (type(prior) == type (None)):
            prior = Vil.Prior(0.5, np.log(0.1),np.log(0.5))
        prior =  prior.get_standarized_Prior(in_features)
        self.prior = prior 
        
        """
        Mean and rhos of the parameters
        """
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))# , requires_grad=True
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.mu_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('rho_bias', None)
            self.register_parameter('mu_bias', None)
            
        """
        The sampled weights
        """
        self.weight = torch.Tensor(out_features, in_features)
        if bias:
            self.bias = torch.Tensor(out_features,1)
        else:
            self.register_parameter('bias', None)
        
        if(0):
            print ("linear bias device: ",self.bias.device)
            print ("linear weights device: ",self.weight.device)
            print ("linear bias mu device: ",self.mu_bias.device)
            print ("linear bias rho device: ",self.rho_bias.device)
            
            print ("linear weights mu  device: ",self.mu_weight.device)
            print ("linear weights rho device: ",self.rho_weight.device)
            
        ## Initialize the Variational variables
        self.reset_parameters()
        self.sample_posterior()
    
    def reset_parameters(self):
        """
        In this function we initialize the parameters using the prior.
        The variance of the weights depends on the prior !! 
        TODO: Should it depend on dimensionality ! 
        Also the initializaion of weights should follow the normal scheme or from prior ? Can they be different
        """
        print ("mu_bias prior LinearVB: ", self.prior.mu_bias)
        self.rho_weight.data = Vil.init_rho(self.mu_weight.size(), self.prior)
        if self.bias is not None:
            self.rho_bias.data = Vil.init_rho(self.mu_bias.size(), self.prior)
        
        ## Now initialize the mean
        self.mu_weight.data = Vil.init_mu(self.mu_weight.size(), self.prior,Ninput = self.mu_weight.size(1))
        if self.bias is not None:
            self.mu_bias.data = Vil.init_mu(self.mu_bias.size(), self.prior, Ninput = self.mu_weight.size(1))

    def sample_posterior(self):
        """
        This function samples the Bayesian weights from the parameters and puts them into the variables.
        It needs to do so using the reparametrization trick so that we can derive respect to sigma and mu
        """
        
#        print ("SAMPLING FROM LINEAR VB")
        if (self.posterior_mean == False):
            
            self.weight = Vil.sample_posterior(self.mu_weight, Vil.softplus(self.rho_weight))
            if self.bias is not None:
                self.bias = Vil.sample_posterior(self.mu_bias, Vil.softplus(self.rho_bias))
        else:
            self.weight.data = self.mu_weight.data
            if self.bias is not None:
                self.bias.data = self.mu_bias.data
        
    def get_KL_divergence(self):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        It does not sample the weights again, it uses the ones that are already sampled.
        
        """
        KL_loss_W = Vil.get_KL_divergence_Samples(self.mu_weight, Vil.softplus(self.rho_weight),
                                                  self.weight, self.prior, mu_prior_fluid = self.prior.mu_weight)
        KL_loss_b = 0
        if self.bias is not None:
            KL_loss_b = Vil.get_KL_divergence_Samples(self.mu_bias, Vil.softplus(self.rho_bias), 
                                                      self.bias,  self.prior, mu_prior_fluid = self.prior.mu_bias)
            
        KL_loss = KL_loss_W + KL_loss_b
        
        return KL_loss
    
    def forward(self, X):
        """
        Funciton call to generate the output, every time we call it, the dynamic graph is created.
        There can be difference between forward in training and test:
            - In dropout we do not zero neurons in test
            - In Variational Inference we dont randombly sample from the posterior
        
        We create the forward pass by performing operations between the input X (Nsam_batch, Ndim)
        and the parameters of the model that we should have initialized in the __init__
        """
        
#        o2 = torch.mm(X, self.weight) + self.bias
        o2 = F.linear(X, self.weight, self.bias)
        return o2
    
    """
    Flag to set that we actually get the posterior mean and not a sample from the random variables
    """
    def set_posterior_mean(self, posterior_mean):
        self.posterior_mean = posterior_mean
        

    
