# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from GeneralVBModelRNN import GeneralVBModelRNN

import Variational_inferences_lib as Vil
from LinearVB import LinearVB 

import string

    
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def letterToIndex(letter):
    return all_letters.find(letter)
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
        return tensor
class RNN_names_classifier(GeneralVBModelRNN):
    """
    This model is intended to host hybrid combinations of VB models and normal ones.
    It incorporates:
        - Main configuration files conf_a where you have everyo
        - Loading and saving parameter from disk.
        - Easy support for VB using the LinearVB parameters.
    
    
    """
    def __init__(self, conf_a, prior):
#        super(RNN_names_classifier, self).__init__(conf_a, prior)
        torch.nn.Module.__init__(self)
        self.loss_func = conf_a.loss_func
        self.lr = conf_a.lr
        self.cf_a = conf_a
        self.prior = prior
        self.future = 0
        self.weight_classes = None 
        ## Use the linear model NN given by pyTorch that already does all the initialization
        ## and everything
        
        self.lstm1 = nn.LSTMCell(conf_a.D_in, conf_a.HS).to(device=conf_a.device, dtype=conf_a.dtype)
#        self.lstm2 = nn.LSTMCell(conf_a.HS, conf_a.HS).to(device=conf_a.device, dtype=conf_a.dtype)
#        self.linear = nn.Linear(conf_a.HS, 1).to(device=conf_a.device, dtype=conf_a.dtype)
        self.linear = LinearVB(in_features = conf_a.HS, out_features = conf_a.D_out, bias=True, prior = prior).to(device=conf_a.device, dtype=conf_a.dtype)
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
    
    def forward(self, X_list_of_chains):
        
        """
        X is a list of tensors from which to evaluate the performance.
        Every element in X can have any length.
        The batch size is 1 in this case... we just run it a number times
        
        """
        self.sample_posterior()

#        print ("Total_sample_dim", X.shape)
        h_t = torch.zeros(X_list_of_chains[0].size(1), self.cf_a.HS, dtype=self.cf_a.dtype, device = self.cf_a.device)
        c_t = torch.zeros(X_list_of_chains[0].size(1),  self.cf_a.HS, dtype=self.cf_a.dtype,  device = self.cf_a.device)

        ## We generate the output for every vector in the chain
        outputs = []
        for X in X_list_of_chains:
            for i, input_t in enumerate(X.chunk(X.size(0), dim=0)):
                input_t = input_t[:,0,:]
#                print ("One_timestep_dim",input_t.shape)
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.cat(outputs, 0)
#        print ("prediction dim ", output.shape)
#        print ("predictions dim ", outputs.shape)
        return outputs
    
    def set_languages(self,languages):
        self.languages = np.array(languages)


    def get_line_language(self, line = "Hola"):
        X_i = lineToTensor(line).to(device = self.cf_a.device, dtype = self.cf_a.dtype)
        return self.predict_language([X_i])
    
    def get_languages_probabilities(self, line = "Hola"):
        X_i = lineToTensor(line).to(device = self.cf_a.device, dtype = self.cf_a.dtype)
        probabilities = self.predict_proba([X_i])
        
        return probabilities
    
    def predict_language(self,X):
        return self.languages[self.predict(X)]
    
    def get_confusion_matrix(self,X,Y):
        Nclasses = self.languages.size
        confusion = np.zeros((Nclasses,Nclasses))
        predictions = self.predict(X)
        
        for i in range(len(X)):  # For each prediction
            confusion[Y[i], predictions[i]] += 1
            
        for i in range(Nclasses):
            confusion[i] = confusion[i] / confusion[i].sum()
    
        return confusion
    
    def set_imbalances(self,Ytrain):
        """
        Set imbalances for training in the samples
        """
        w = np.zeros((self.D_out,1))
        for i in range(self.D_out):
            w[i] = np.sum(Ytrain.numpy() == i)
        
        self.weight_classes = w/np.sum(w)
        
        
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
    

