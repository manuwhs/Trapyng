# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

import Variational_inferences_lib as Vil

class HalfBayesianMLP(nn.Module):
    
    def __init__(self, conf_a, prior):
        super(HalfBayesianMLP, self).__init__()
        
        self.loss_func = conf_a.loss_func
        self.lr = conf_a.lr
        self.cf_a = conf_a
        self.prior = prior
        
        ## Use the linear model NN given by pyTorch that already does all the initialization
        ## and everything
        
        self.linear1 = torch.nn.Linear(in_features = conf_a.D_in, out_features = conf_a.H, bias=True)
        
#        self.W1 = Parameter(torch.randn(conf_a.D_in, conf_a.H, device=device, dtype=dtype, requires_grad=True))
#        self.b1 = Parameter(torch.randn(1, conf_a.H, device=device, dtype=dtype, requires_grad=True))
        
        # The second layer are parameters that we define.
        # We need to use the Parameter() function so that the parameters are internally
        # associated to our module
        
        """
        Create the Bayesian Parameters and the weight variables. 
        For each weight the variables are:
            - mu: The mean value of the 
            - rho: The sigma of th OR sigma
        """
        self.mu_W2 = Parameter(torch.zeros(conf_a.H, conf_a.D_out, device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        self.mu_b2 = Parameter(torch.zeros(1, conf_a.D_out, device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))

        W2_size = (conf_a.H, conf_a.D_out)
        b2_size = (1, conf_a.D_out)
        # INITIALIZE !! Real parameters that will be used for computing the output of the network
#        self.W2 = Vil.initialization_sample_posterior(self.mu_W2.size(), self.prior)
#        self.b2 = Vil.initialization_sample_posterior(self.mu_b2.size(),self.prior)
        
#        self.sigma_W2 = Parameter(torch.tensor(Vil.initialization_sigmas_posteior(self.mu_W2.size(), self.prior), device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
#        self.sigma_b2 = Parameter(torch.tensor(Vil.initialization_sigmas_posteior(self.mu_b2.size(), self.prior), device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        
        self.rho_W2 = Parameter(torch.tensor(Vil.initialization_rhos_posteior(W2_size, self.prior), device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        self.rho_b2 = Parameter(torch.tensor(Vil.initialization_rhos_posteior(b2_size, self.prior), device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        
        ## Initilize the random variables !! 
        self.mu_W2 =  Parameter(torch.tensor(Vil.sample_posterior(self.mu_W2, Vil.softplus(self.rho_W2), False), device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        self.mu_b2 =  Parameter(torch.tensor(Vil.sample_posterior(self.mu_b2, Vil.softplus(self.rho_b2), False), device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        
    def sample_posterior(self, posterior_mean = False):
        """
        This function samples the Bayesian weights from the parameters and puts them into the variables.
        It needs to do so using the reparametrization trick so that we can derive respect to sigma and mu
        """
        
        self.W2 = Vil.sample_posterior(self.mu_W2, Vil.softplus(self.rho_W2), posterior_mean)
        self.b2 = Vil.sample_posterior(self.mu_b2, Vil.softplus(self.rho_b2), posterior_mean)
        
    def compute_KL_loss(self):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        
        """
        KL_loss_W2 = Vil.get_KL_divergence_Samples(self.mu_W2, Vil.softplus(self.rho_W2), self.W2, self.prior)
        KL_loss_b2 = Vil.get_KL_divergence_Samples(self.mu_b2, Vil.softplus(self.rho_b2), self.b2,  self.prior)
        KL_loss = KL_loss_W2 + KL_loss_b2
        
        return KL_loss
    
    def forward(self, X, posterior_mean = False):
        """
        Funciton call to generate the output, every time we call it, the dynamic graph is created.
        There can be difference between forward in training and test:
            - In dropout we do not zero neurons in test
            - In Variational Inference we dont randombly sample from the posterior
        
        We create the forward pass by performing operations between the input X (Nsam_batch, Ndim)
        and the parameters of the model that we should have initialized in the __init__
        """
        
        ## We need to sample from the posterior !! 
        self.sample_posterior(posterior_mean)
        
        o1 = self.linear1(X)
#        o1 = torch.mm(X, self.W1) + self.b1
#        print ("x shape: ", X.shape, "W1 shape: ", self.W1.shape, "b1 shape: ", self.b1.shape)
#        print ("o1 shape: ", o1.shape)
#        print ("W2 shape: ", self.W2.shape, "b2 shape: ", self.b2.shape)
        
        ## Apply non-linearity
        o1 = self.cf_a.activation_func(o1)
        o1 = F.dropout(o1,p = self.cf_a.dop, training = self.training)
        o2 = torch.mm(o1, self.W2) + self.b2
#        print ("o2 shape: ", o2.shape)
        return o2
    
    """
    ################################# SAVE AND LOAD MODEL ####################
    """
    
    def save(self, path):
        """
        This function saves all the parameters and states of the model.
        Some tailoring have to be made depending on what we want to save and load.
        We need to save:
            - The paramters of the model 
            - 
        """
        print ("Storing sate dict in file: ",path )
        torch.save(self.state_dict(), path)
         
    def load(self, path):
        """
        This function loads all the parameters and states of the model.
        """
        print ("Loading sate dict from file: ",path )
        self.load_state_dict(torch.load(path))
    
    """
    ############################### INTERFACE FUNCTIONS ###########################
    """
    
    def predict(self, X, posterior_mean = False):
        """ sklearn interface without creating graph """
        with torch.no_grad():
            return self.forward(X, posterior_mean)
        
    def get_loss(self, X, Y, posterior_mean = False):
        """ sklearn interface without creating graph """
        with torch.no_grad():
            return self.loss_func(self.forward(X,posterior_mean),Y)
        
    """
    ############################### Training FUNCTIONS ###########################
    """
    
    def set_Bayesian_requires_grad(self, requires_grad = True):
        self.mu_W2.requires_grad = requires_grad;
        self.rho_W2.requires_grad = requires_grad;
        self.mu_b2.requires_grad = requires_grad;
        self.rho_b2.requires_grad = requires_grad;
        
    def set_NonBayesian_requires_grad(self, requires_grad = True):
        self.linear1.weight.requires_grad = requires_grad
        self.linear1.bias.requires_grad = requires_grad
        
    
    def get_final_loss(self,X,Y,posterior_mean = True):
        predictions = self.forward(X,posterior_mean)
        
        # Now we can just compute both losses which will build the dynamic graph
        loss_normal_weights = self.loss_func(predictions, Y)
        loss_varitional_weights =  loss_normal_weights  # - 0.001* self.compute_KL_loss()
        
        return loss_varitional_weights
    
    def train_batch_optimi_2(self, X_batch, Y_batch):
        """
        If we want to optimize the weights, each with different loss functions
        """
        
        # First we compute the predictions, creating all the network for
        # all learnable parameters
        predictions = self.forward(X_batch)
        
        # Now we can just compute both losses which will build the dynamic graph
        loss_normal_weights = self.loss_func(predictions, Y_batch)
        loss_varitional_weights =  loss_normal_weights  #  - 0.001* self.compute_KL_loss()
#        loss2 = loss
        
        self.zero_grad()     # zeroes the gradient buffers of all parameters
        
        """
        First lets compute the gradients for everyone !!
        We start by the normal weights, and then the Bayesian, so we deactivate 
        the Bayesian and activate the 
        """
        self.set_Bayesian_requires_grad(False)
        self.set_NonBayesian_requires_grad(True)
        loss_normal_weights.backward(retain_graph = True)  # retain_graph = False
        
        
        self.set_Bayesian_requires_grad(True)
        self.set_NonBayesian_requires_grad(False)
        loss_varitional_weights.backward(retain_graph = False)  # retain_graph = False
        
        self.set_NonBayesian_requires_grad(True) # Set the last ones !!
        
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        with torch.no_grad():
            for f in parameters:
                f.data.sub_(f.grad.data * self.lr)
        return loss_normal_weights
    
    """
    ############################### PRINT FUNCTIONS ###########################
    """
    
    def print_parameters_names(self):
        pass
    
    def print_named_parameters(self):
        print ("--------- NAMED PARAMETERS ------------")
        for f in self.named_parameters():
            print ("Name: ",f[0])
            print (f[1])

    def print_parameters(self):
        print ("--------- PARAMETERS ------------")
        for f in self.parameters():
            print (f.data)
    
    def print_gradients(self, X, Y):
        """
        Print the gradients between the output and X
        """
        print ("--------- GRADIENTS ------------")
        predictions = self.forward(X)
        
        ## Define the loss: 
        loss = torch.sum(torch.pow(predictions - Y, 2))
        
        ## Clean previous gradients 
        self.zero_grad()
        loss.backward()
        
        print (self.linear1.weight.grad)
        print (self.linear1.bias.grad)
        
        print (self.W2.grad)
        print (self.b2.grad)
        
        print ("----------- STRUCTURE ------------")
        ## Clean previous gradients 
        print(loss.grad_fn)                       # MSELoss
        print(loss.grad_fn.next_functions[0][0])  # Linear 1
        print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # Sigmoid
    
    
        self.zero_grad()
