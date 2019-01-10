# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


class BasicMLP(nn.Module):
    
    def __init__(self, conf_a):
        super(BasicMLP, self).__init__()
        self.loss_func = conf_a.loss_func
        self.lr = conf_a.lr
        self.cf_a = conf_a
        
        ## Use the linear model NN given by pyTorch that already does all the initialization
        ## and everything
        
        self.linear1 = torch.nn.Linear(in_features = conf_a.D_in, out_features = conf_a.H, bias=True)
        
#        self.W1 = Parameter(torch.randn(conf_a.D_in, conf_a.H, device=device, dtype=dtype, requires_grad=True))
#        self.b1 = Parameter(torch.randn(1, conf_a.H, device=device, dtype=dtype, requires_grad=True))
        
        # The second layer are parameters that we define.
        # We need to use the Parameter() function so that the parameters are internally
        # associated to our module
        
        self.W2 = Parameter(torch.randn(conf_a.H, conf_a.D_out, device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        self.b2 = Parameter(torch.randn(1, conf_a.D_out, device= conf_a.device, dtype= conf_a.dtype, requires_grad=True))
        ## Initialize with Xavier !!
        self.W2 = torch.nn.init.xavier_normal_(self.W2, gain=1)
        self.b2 = torch.nn.init.xavier_normal_(self.b2, gain=1)
        
    def forward(self, X):
        """
        Funciton call to generate the output, every time we call it, the dynamic graph is created.
        There can be difference between forward in training and test:
            - In dropout we do not zero neurons in test
            - In Variational Inference we dont randombly sample from the posterior
        
        We create the forward pass by performing operations between the input X (Nsam_batch, Ndim)
        and the parameters of the model that we should have initialized in the __init__
        """
        
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
    
    def predict(self, X):
        """ sklearn interface without creating graph """
        with torch.no_grad():
            return self.forward(X)
        
    def get_loss(self, X, Y):
        """ sklearn interface without creating graph """
        with torch.no_grad():
            return self.loss_func(self.forward(X),Y)
        
    """
    ############################### Training FUNCTIONS ###########################
    """
    
    def train_batch(self, X_batch, Y_batch):
        predictions = self.forward(X_batch)
        loss = self.loss_func(predictions, Y_batch)
    
        # loss2 = torch.mean(torch.log(torch.pow(output - target,4))) /1000;
        self.zero_grad()     # zeroes the gradient buffers of all parameters
        loss.backward(retain_graph = False)
        
        """
        Update all the parameters.
        We stop creating the the dynamic chart.
        """
        with torch.no_grad():
            ## Train the parameters set as trainable !! 
            parameters = filter(lambda p: p.requires_grad, self.parameters())
            for f in parameters:
                f.data.sub_(f.grad.data * self.lr)
        
        return loss
    
    def train_batch_optimi(self, X_batch, Y_batch, together = True):
        """
        If we have assigned different optimizers to each weights
        """
        predictions = self.forward(X_batch)
        loss = self.loss_func(predictions, Y_batch)
#        loss2 = torch.mean(torch.log(torch.pow(output - target,4))) /1000;
        # loss2 = torch.mean(torch.log(torch.pow(output - target,4))) /1000;
        
        if (together):
            optimizers = [self.cf_a.op_a]
        else:
            optimizers = [self.cf_a.op_h,self.cf_a.op_o ]
            
        for optimizer in optimizers:    
            optimizer.zero_grad()   # zero the gradient buffers
            
        loss.backward()  # retain_graph = False
        
        for optimizer in optimizers:    
            optimizer.step()    # Does the update
        
        return loss
    
    def train_batch_optimi_2(self, X_batch, Y_batch):
        """
        If we want to optimize the weights, each with different loss functions
        """
        
        # First we compute the predictions, creating all the network for
        # all learnable parameters
        predictions = self.forward(X_batch)
        
        # Now we can just compute both losses which will build the dynamic graph
        loss_h = self.loss_func(predictions, Y_batch)
        loss_o = torch.mean(torch.pow(predictions - Y_batch,2))/1.2;
#        loss2 = loss

        self.zero_grad()     # zeroes the gradient buffers of all parameters
        
        # Now we want to compute the gradients for just the outout layer
    
        """
        The wrapper "with torch.no_grad()" temporarily set all the requires_grad flag to false. 
        """
        if(0):
            with torch.no_grad():
                self.W2.requires_grad = True
                self.b2.requires_grad = True
                loss_o.backward(retain_graph = True)  # retain_graph = False
                print (self.linear1.weight.requires_grad )
    #        self.zero_grad()     # zeroes the gradient buffers of all parameters
            with torch.no_grad():
                self.linear1.weight.requires_grad = True
                self.linear1.bias.requires_grad = True
                loss_h.backward()  # retain_graph = False
                
            with torch.no_grad():
                ## Train the parameters set as trainable !! 
                parameters = filter(lambda p: p.requires_grad, self.parameters())
                for f in parameters:
                    f.data.sub_(f.grad.data * self.lr)
        else:

            self.W2.requires_grad = True
            self.b2.requires_grad = True
            self.linear1.weight.requires_grad = False
            self.linear1.bias.requires_grad = False
            
            """
            We need to use retain_graph = True so that the backward function does not remove 
            the input values computed that are needed to compute the gradients !!! 
            """
            loss_o.backward(retain_graph = True)  # retain_graph = False

#            self.zero_grad()     # zeroes the gradient buffers of all parameters
            with torch.no_grad():
                self.W2.requires_grad = False
                self.b2.requires_grad = False
                self.linear1.weight.requires_grad = True
                self.linear1.bias.requires_grad = True
                
                loss_h.backward()  # retain_graph = False
            
            
            self.W2.requires_grad = True
            self.b2.requires_grad = True
            self.linear1.weight.requires_grad = True
            self.linear1.bias.requires_grad = True
            ## Train the parameters set as trainable !! 
            parameters = filter(lambda p: p.requires_grad, self.parameters())
            with torch.no_grad():
                for f in parameters:
                    f.data.sub_(f.grad.data * self.lr)
        return loss_h
    
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
