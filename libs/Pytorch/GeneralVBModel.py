# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

import Variational_inferences_lib as Vil
from LinearVB import LinearVB 
import pyTorch_utils as pytut

class GeneralVBModel(nn.Module):
    """
    This model is intended to host hybrid combinations of VB models and normal ones.
    It incorporates:
        - Main configuration files conf_a where you have everyo
        - Loading and saving parameter from disk.
        - Easy support for VB using the LinearVB parameters.
    
    In this model one should:
        - Instantiate the optimizer as _optimizer using the conf_a parameters
        - Instantiate the Priors from the parameters.
    
    """
    def __init__(self, conf_a):
        super(GeneralVBModel, self).__init__()
        
        self.loss_func = conf_a.loss_func
        self.cf_a = conf_a
        
        ## Use the linear model NN given by pyTorch that already does all the initialization
        ## and everything
        
        self.linear1 = LinearVB(in_features = conf_a.D_in, 
                                out_features = conf_a.H, bias=True, prior = Vil.Prior(**conf_a.input_layer_prior))
        self.linear2 = LinearVB(conf_a.H, out_features = conf_a.D_out, 
                                bias=True, prior =  Vil.Prior(**conf_a.output_layer_prior))
        
        
#        self.linear2 = torch.nn.Linear(in_features = conf_a.H, out_features = conf_a.D_out, bias=True)

        
        """
        List of Bayesian Linear Models.
        Using this list we can easily set the special requirements of VB models.
        And also analize easily the weights in the network
        """
        self.VBmodels = [self.linear1, self.linear2]
        self.LinearModels = [] # self.linear2
        
        optimizer = pytut.get_optimizers(self, self.cf_a)
        self._optimizer = optimizer
    """
    ############################### SPECIAL VB #################################
    """
    
    def sample_posterior(self):
        """
        This function samples the Bayesian weights from the parameters and puts them into the variables.
        It needs to do so using the reparametrization trick so that we can derive respect to sigma and mu
        """
        
        for VBmodel in self.VBmodels:
            VBmodel.sample_posterior()
        
    def get_KL_divergence(self):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        """
        
        KL_loss = 0 #.to(dtype = self.cf_a.dtype, device =self.cf_a.device )
        for VBmodel in self.VBmodels:
            KL_loss += VBmodel.get_KL_divergence()
            
        return KL_loss

    def set_posterior_mean(self, posterior_mean):
        """
        Set the Bayesian Models to be either sampling or getting the most likely samples.
        
        """
        for VBmodel in self.VBmodels:
            VBmodel.set_posterior_mean(posterior_mean)
            
    def combine_losses(self, data_loss, KL_divergence):
        KL_constant = self.cf_a.eta_KL * 1/(self.cf_a.Nsamples_train/self.cf_a.batch_size_train);
#        print ("KL constant: ", KL_constant)
        return data_loss  + KL_constant * KL_divergence
    
    def forward(self, X):
        """
        Funciton call to generate the output, every time we call it, the dynamic graph is created.
        There can be difference between forward in training and test:
            - In dropout we do not zero neurons in test
            - In Variational Inference we dont randombly sample from the posterior
        
        We create the forward pass by performing operations between the input X (Nsam_batch, Ndim)
        and the parameters of the model that we should have initialized in the __init__
        """
        ## We need to sample from the posterior !! 
        self.sample_posterior()
        
#        o = self.linear1(X)
        o = self.linear1(X)
        ## Apply non-linearity
        o = self.cf_a.activation_func(o)
        
        o = F.dropout(o,p = self.cf_a.dop, training = self.training)
        
        o = self.linear2(o)
#        print ("o2 shape: ", o2.shape)
        return o
    
    """
    ############################### INTERFACE FUNCTIONS ###########################
    sklearn interface without creating graph
    """
    
    def predict(self, X):
        """ sklearn interface without creating graph """
        X = X.to(device =self.cf_a.device )
        if (self.cf_a.task_type == "regression"):
            with torch.no_grad():
                return self.forward(X)
        elif(self.cf_a.task_type == "classification"):
            with torch.no_grad():
                return torch.argmax(self.forward(X),1)
     
    def predict_proba(self,X):
        X = X.to(device =self.cf_a.device )
        
        if (self.cf_a.task_type == "regression"):
            with torch.no_grad():
                return self.forward(X)
        elif(self.cf_a.task_type == "classification"):
            with torch.no_grad():
                return  nn.functional.softmax(self.forward(X), dim = 1)
            
    
    def get_data_loss(self, X,Y):
        """
        The loss of the data.
        TODO: Should I not create the graph here ?
        """
        X = X.to(device =self.cf_a.device )
        Y = Y.to(device =self.cf_a.device )
#        print ("Size of Y", Y.shape, " Y ", Y)
#        print ("Size of predictions", self.forward(X).shape)
        with torch.no_grad():
            return self.loss_func(self.forward(X),Y)

    def get_loss(self, X, Y):
        """ 
        Data Loss + VB loss
        """
        X = X.to(device =self.cf_a.device )
        Y = Y.to(device =self.cf_a.device )
        with torch.no_grad():
            # Now we can just compute both losses which will build the dynamic graph
            data_loss = self.get_data_loss(X, Y)
            KL_div = self.get_KL_divergence()
            total_loss =  self.combine_losses(data_loss, KL_div)
        
        return total_loss
    
    def get_KL_loss(self):
        """
        Computes the KL div but without creating a graph !!
        """
        with torch.no_grad():
            KL_div = self.get_KL_divergence();
        return KL_div
    
    """
    ############################### Training FUNCTIONS ###########################
    """
    
    def set_Bayesian_requires_grad(self, requires_grad = True):
        
        for VBmodel in self.VBmodels:
            VBmodel.rho_weight.requires_grad = requires_grad;
            VBmodel.mu_weight.requires_grad = requires_grad;
            VBmodel.rho_bias.requires_grad = requires_grad;
            VBmodel.mu_bias.requires_grad = requires_grad;
        
    def set_NonBayesian_requires_grad(self, requires_grad = True):
        for LinearModel in self.LinearModelS:
        
            LinearModel.weight.requires_grad = requires_grad
            LinearModel.bias.requires_grad = requires_grad
        
    def train_batch(self, X_batch, Y_batch):
        """
        It is enough to just compute the total loss because the normal weights 
        do not depend on the KL Divergence
        """
        # Now we can just compute both losses which will build the dynamic graph
        predictions = self.forward(X_batch)
        data_loss = self.loss_func(predictions,Y_batch)
        KL_div = self.get_KL_divergence()
        total_loss =  self.combine_losses(data_loss, KL_div)

        self.zero_grad()     # zeroes the gradient buffers of all parameters
        total_loss.backward()
        
        if (type(self._optimizer) == type(None)):
            
            self.zero_grad()     # zeroes the gradient buffers of all parameters
            total_loss.backward()
            parameters = filter(lambda p: p.requires_grad, self.parameters())
            with torch.no_grad():
                for f in parameters:
                    f.data.sub_(f.grad.data * self.lr )
        else:
#            print ("Training")
            self._optimizer.step()
            self._optimizer.zero_grad()
        return total_loss
    
    def train_batch2(self, X_batch, Y_batch):
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
    ############################### PRINT FUNCTIONS ###########################
    """
    
    def print_parameters_names(self):
        pass
    
    def print_named_parameters(self):
        print ("--------- NAMED PARAMETERS ------------")
        for f in self.named_parameters():
            print ("Name: ",f[0])
            print ("requires_grad: ", f[1].requires_grad)
            print (f[1])

    def print_parameters(self):
        print ("--------- PARAMETERS ------------")
        for f in self.named_parameters():
            print ("Name: ",f[0])
            print ("requires_grad: ", f[1].requires_grad)
            print (f[1])
    
    def print_gradients(self, X, Y):
        """
        Print the gradients between the output and X
        """
        print ("--------- GRADIENTS ------------")
        predictions = self.forward(X)
        
        ## Define the loss: 
        loss = self.loss_func(predictions,Y)
        
        ## Clean previous gradients 
        self.zero_grad()
        loss.backward()

        if (len(self.VBmodels) >0):
            print ("#### Bayesian Gradients ####")
            for VBmodel in self.VBmodels:
                for f in VBmodel.named_parameters():
                    print ("Name: ",f[0])
                    print (f[1].grad)
                
        if (len(self.LinearModels) >0):
            print ("#### Normal Weight Gradients ####")
                   
            for LinearModel in self.LinearModels:
                print(LinearModel.weight.grad)
                print(LinearModel.bias.grad)
            
        
        print ("----------- STRUCTURE ------------")
        ## Clean previous gradients 
        print(loss.grad_fn)                       # MSELoss
        print(loss.grad_fn.next_functions[0][0])  # Linear 1
        print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # Sigmoid
    
    
        self.zero_grad()
