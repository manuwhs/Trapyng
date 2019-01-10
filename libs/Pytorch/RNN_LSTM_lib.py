
"""
Library that uses the LSTMCell or LSTM to create RNN.
Similar to RNNBase in Pyotrch but using LSTMCells
"""
import math
import torch
from torch.nn.parameter import Parameter
import Variational_inferences_lib as Vil
from torch.nn.modules.rnn import RNNCellBase,RNNBase
import numpy as np

from LSTMCellVB import LSTMCellVB

class RNN_LSTM(RNNCellBase):
    
    def __init__(self, input_size, hidden_size,
                     num_layers=1, bias=True, batch_first=False,
                     dropout=0, bidirectional=False, Bayesian = True, prior = None):
        
        super(RNN_LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.Bayesian = Bayesian
        if (type(prior) == type (None)):
            prior = Vil.Prior(0.5, np.log(0.1),np.log(0.5))
        self.prior = prior
        
        if (self.Bayesian):
            for i in range(num_layers):
                LSTMCell = LSTMCellVB(input_size = input_size, hidden_size = hidden_size, bias = bias, prior = prior)
                setattr(self, 'LSTMCell%i'%(i+1),LSTMCell ) 
                input_size =  hidden_size # For the subsequent layers
        else:
            for i in range(num_layers):
                LSTMCell = torch.nn.LSTMCell(input_size = input_size, hidden_size = hidden_size, bias = bias)
                setattr(self, 'LSTMCell%i'%(i+1),LSTMCell )
                input_size =  hidden_size # For the subsequent layers
                
    def get_LSTMCells(self):
        list_cells = []
        for i in range(self.num_layers):
            list_cells.append(getattr(self, 'LSTMCell%i'%(i+1)))
        
        return list_cells
    
    def reset_parameters(self):
        """
        In this function we initialize the parameters using the prior.
        The variance of the weights depends on the prior !! 
        TODO: Should it depend on dimensionality ! 
        Also the initializaion of weights should follow the normal scheme or from prior ? Can they be different
        """
        
        for i in range(self.num_layers):
            getattr(self, 'LSTMCell%i'%(i+1)).reset_parameters()
    
    
    def sample_posterior(self):
        """
        This function samples the Bayesian weights from the parameters and puts them into the variables.
        It needs to do so using the reparametrization trick so that we can derive respect to sigma and mu
        """
        if(self.Bayesian):
            for i in range(self.num_layers):
                getattr(self, 'LSTMCell%i'%(i+1)).sample_posterior()
                
    def get_KL_divergence(self):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        It does not sample the weights again, it uses the ones that are already sampled.
        
        """
        KL_loss = 0
        if(self.Bayesian):
            for i in range(self.num_layers):
                KL_loss += getattr(self, 'LSTMCell%i'%(i+1)).get_KL_divergence()
                
        return KL_loss

    """
    Flag to set that we actually get the posterior mean and not a sample from the random variables
    """
    def set_posterior_mean(self, posterior_mean):
        if(self.Bayesian):
            for i in range(self.num_layers):
                getattr(self, 'LSTMCell%i'%(i+1)).set_posterior_mean()
                
    def forward(self, X, hx=None):
        """
        Assuming batch_first the dimensions are (batch_sie, Nseq, Dim)
        """
        outputs = []
        (batch_size, Nseq, Dim) = X.size()
        
#        print ("Dimensions LSTM_RNN Batch: ", (batch_size, Nseq, Dim))
        for l in range(self.num_layers):
            if (type(hx) == type(None)):
                h_t = torch.zeros(batch_size, self.hidden_size).to(device = Vil.device, dtype = Vil.dtype)
                c_t = torch.zeros(batch_size,  self.hidden_size).to(device = Vil.device, dtype = Vil.dtype)
            else:
                h_t = torch.zeros(batch_size, self.hidden_size).to(device = Vil.device, dtype = Vil.dtype)
                c_t = torch.zeros(batch_size,  self.hidden_size).to(device = Vil.device, dtype = Vil.dtype)
            setattr(self, 'h_t%i'%(l+1), h_t)
            setattr(self, 'c_t%i'%(l+1), c_t)
            
        # We loop for every element in the chain and for every layer

        for i in range(Nseq):
            input_t = X[:,i,:]
#            print ("Sequence Chunk size: ",input_t.size())
            l = 0
            # First layer we put the input, in the rest we put the propagated states
            h_t, c_t = getattr(self, 'LSTMCell%i'%(l+1))(input_t, (getattr(self, 'h_t%i'%(l+1)), getattr(self, 'c_t%i'%(l+1))))
            setattr(self, 'h_t%i'%(l+1), h_t)
            setattr(self, 'c_t%i'%(l+1), c_t)
            
            for l in range(1,self.num_layers):
                h_t, c_t = getattr(self, 'LSTMCell%i'%(l+1))(h_t, (getattr(self, 'h_t%i'%(l+1)), getattr(self, 'c_t%i'%(l+1))))
                setattr(self, 'h_t%i'%(l+1), h_t)
                setattr(self, 'c_t%i'%(l+1), c_t)
            
        
        # Return the hx and cx of all layers ? for the last sample ?
        outputs = []
        for l in range(self.num_layers):
            outputs.append( [getattr(self, 'h_t%i'%(l+1)), getattr(self, 'c_t%i'%(l+1))])

#        outputs = torch.stack(outputs, 1).squeeze(2)
        
        return outputs
        
