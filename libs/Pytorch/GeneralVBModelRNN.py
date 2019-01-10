# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from GeneralVBModel import GeneralVBModel

import Variational_inferences_lib as Vil
from LinearVB import LinearVB 
from LSTMCellVB import LSTMCellVB

class GeneralVBModelRNN(GeneralVBModel):
    """
    This model is intended to host hybrid combinations of VB models and normal ones.
    It incorporates:
        - Main configuration files conf_a where you have everyo
        - Loading and saving parameter from disk.
        - Easy support for VB using the LinearVB parameters.
    
    
    """
    def __init__(self, conf_a, prior):
#        super(GeneralVBModelRNN, self).__init__()
        torch.nn.Module.__init__(self)
        self.loss_func = conf_a.loss_func
        self.lr = conf_a.lr
        self.cf_a = conf_a
        self.prior = prior
        self.future = 0
        ## Use the linear model NN given by pyTorch that already does all the initialization
        ## and everything
        
        self.lstm1 = nn.LSTMCell(1, conf_a.HS).to(device=conf_a.device, dtype=conf_a.dtype)
        self.lstm2 = nn.LSTMCell(conf_a.HS, conf_a.HS).to(device=conf_a.device, dtype=conf_a.dtype)
#        self.linear = nn.Linear(conf_a.HS, 1).to(device=conf_a.device, dtype=conf_a.dtype)
        self.linear = LinearVB(in_features = conf_a.HS, out_features = 1, bias=True, prior = prior).to(device=conf_a.device, dtype=conf_a.dtype)
#        self.linear2 = LinearVB(in_features = 10, out_features = 1, bias=True, prior = prior).to(device=conf_a.device, dtype=conf_a.dtype)
        
        """
        List of Bayesian Linear Models.
        Using this list we can easily set the special requirements of VB models.
        And also analize easily the weights in the network
        """
        self.VBmodels = [self.linear]
        self.LinearModels = []
    
    def set_future(self, value):
        """
        When calling forward, if future is not 0, then the network will also try to predict the future of the NN.
        This way we do not need to put it as a parameter of forward and we can reuse all the code.
        """
        self.future = value
        
    def forward(self, X):
        """
        In this case we can predict the next 
        """
        self.sample_posterior()
        outputs = []
        h_t = torch.zeros(X.size(0), self.cf_a.HS, dtype=self.cf_a.dtype, device = self.cf_a.device)
        c_t = torch.zeros(X.size(0),  self.cf_a.HS, dtype=self.cf_a.dtype, device = self.cf_a.device)
        h_t2 = torch.zeros(X.size(0),  self.cf_a.HS, dtype=self.cf_a.dtype, device = self.cf_a.device)
        c_t2 = torch.zeros(X.size(0),  self.cf_a.HS, dtype=self.cf_a.dtype, device = self.cf_a.device)

        for i, input_t in enumerate(X.chunk(X.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
#            output = self.linear2(self.cf_a.activation_func(self.linear(h_t2)))
            outputs += [output]
            
        for i in range(self.future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


#    def train_batch(self, X_batch, Y_batch):
#        """
#        It is enough to just compute the total loss because the normal weights 
#        do not depend on the KL Divergence
#        """
#        def closure():
#            self.cf_a.optimizer.zero_grad()
#            predictions = self.forward(X_batch)
#            data_loss = self.loss_func(predictions,Y_batch)
#            KL_div = self.get_KL_divergence()
#            total_loss =  self.combine_losses(data_loss, KL_div)
#            print('total_loss:', total_loss.item())
#            total_loss.backward()
#            return total_loss
#        
#        self.cf_a.optimizer.step(closure)
#        
#        return torch.zeros(1)
    

