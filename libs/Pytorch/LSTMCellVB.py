import math
import torch
from torch.nn.parameter import Parameter
import Variational_inferences_lib as Vil
from torch.nn.modules.rnn import RNNCellBase 
import numpy as np
class LSTMCellVB(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, prior = None):
        super(LSTMCellVB, self).__init__()
        self.type_layer = "LSTM"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.posterior_mean = False # Flag to know if we sample from the posterior mean or we actually sample
        
        ## If no prior is specified we just create it ourselves
        if (type(prior) == type (None)):
            prior = Vil.Prior(0.5, np.log(0.1),np.log(0.5))
        self.prior = prior
        
        """
        Variational Inference Parameters
        """
        self.mu_weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size).to(device = Vil.device, dtype = Vil.dtype))
        self.mu_weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size).to(device = Vil.device, dtype = Vil.dtype))
        self.rho_weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size).to(device = Vil.device, dtype = Vil.dtype))
        self.rho_weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size).to(device = Vil.device, dtype = Vil.dtype))
        if bias:
            self.mu_bias_ih = Parameter(torch.Tensor(4 * hidden_size).to(device = Vil.device, dtype = Vil.dtype))
            self.mu_bias_hh = Parameter(torch.Tensor(4 * hidden_size).to(device = Vil.device, dtype = Vil.dtype))
            self.rho_bias_ih = Parameter(torch.Tensor(4 * hidden_size).to(device = Vil.device, dtype = Vil.dtype))
            self.rho_bias_hh = Parameter(torch.Tensor(4 * hidden_size).to(device = Vil.device, dtype = Vil.dtype))
        else:
            self.register_parameter('mu_bias_ih', None)
            self.register_parameter('mu_bias_hh', None)
            self.register_parameter('rho_bias_ih', None)
            self.register_parameter('rho_bias_hh', None)
        """
        Sampled weights
        """

        self.weight_ih = torch.Tensor(4 * hidden_size, input_size).to(device = Vil.device, dtype = Vil.dtype)
        self.weight_hh = torch.Tensor(4 * hidden_size, hidden_size).to(device = Vil.device, dtype = Vil.dtype)
        if bias:
            self.bias_ih = torch.Tensor(4 * hidden_size).to(device = Vil.device, dtype = Vil.dtype)
            self.bias_hh = torch.Tensor(4 * hidden_size).to(device = Vil.device, dtype = Vil.dtype)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        if(1):
            print ("linear bias_ih device: ",self.bias_ih.device)
            print ("linear weights_ih device: ",self.weight_ih.device)
            print ("linear bias mu_ih device: ",self.mu_bias_ih.device)
            print ("linear bias rho_ih device: ",self.rho_bias_ih.device)
            
            print ("linear weights mu_ih  device: ",self.mu_weight_ih.device)
            print ("linear weights rho_ih device: ",self.rho_weight_ih.device)
            
            print ("linear bias_hh device: ",self.bias_hh.device)
            print ("linear weights_hh device: ",self.weight_hh.device)
            print ("linear bias mu_hh device: ",self.mu_bias_hh.device)
            print ("linear bias rho_hh device: ",self.rho_bias_hh.device)
            
            print ("linear weights mu_hh  device: ",self.mu_weight_hh.device)
            print ("linear weights rho_hh device: ",self.rho_weight_hh.device)
            
        self.reset_parameters()
        self.sample_posterior()
        
    def reset_parameters(self):
        """
        In this function we initialize the parameters using the prior.
        The variance of the weights depends on the prior !! 
        TODO: Should it depend on dimensionality ! 
        Also the initializaion of weights should follow the normal scheme or from prior ? Can they be different
        """
        
        self.rho_weight_ih.data = Vil.init_rho(self.rho_weight_ih.size(), self.prior)
        self.rho_weight_hh.data = Vil.init_rho(self.rho_weight_hh.size(), self.prior)
        if self.bias is not None:
            self.rho_bias_ih.data = Vil.init_rho(self.rho_bias_ih.size(), self.prior)
            self.rho_bias_hh.data = Vil.init_rho(self.rho_bias_hh.size(), self.prior)
            
        ## Now initialize the mean
        self.mu_weight_ih.data = Vil.init_mu(self.mu_weight_ih.size(), self.prior,Ninput = self.mu_weight_ih.size(1))
        self.mu_weight_hh.data = Vil.init_mu(self.mu_weight_hh.size(), self.prior,Ninput = self.mu_weight_hh.size(1))
        if self.bias is not None:
            self.mu_bias_ih.data = Vil.init_mu(self.mu_bias_ih.size(), self.prior, Ninput = self.mu_weight_ih.size(1))
            self.mu_bias_hh.data = Vil.init_mu(self.mu_bias_hh.size(), self.prior, Ninput = self.mu_weight_hh.size(1))

    def sample_posterior(self):
        """
        This function samples the Bayesian weights from the parameters and puts them into the variables.
        It needs to do so using the reparametrization trick so that we can derive respect to sigma and mu
        """
        if (self.posterior_mean == False):
            self.weight_ih = Vil.sample_posterior(self.mu_weight_ih, Vil.softplus(self.rho_weight_ih))
            self.weight_hh = Vil.sample_posterior(self.mu_weight_hh, Vil.softplus(self.rho_weight_hh))
            if self.bias is not None:
                self.bias_ih = Vil.sample_posterior(self.mu_bias_ih, Vil.softplus(self.rho_bias_ih))
                self.bias_hh = Vil.sample_posterior(self.mu_bias_ih, Vil.softplus(self.rho_bias_hh))     
        else:
            self.weight_ih.data = self.mu_weight_ih.data
            self.weight_hh.data = self.mu_weight_hh.data
            if self.bias is not None:
                self.bias_hh.data = self.mu_bias_hh.data
                self.bias_ih.data = self.mu_bias_ih.data
                
    def get_KL_divergence(self):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        It does not sample the weights again, it uses the ones that are already sampled.
        
        """
        KL_loss_ih = Vil.get_KL_divergence_Samples(self.mu_weight_ih, Vil.softplus(self.rho_weight_ih), self.weight_ih, self.prior)
        KL_loss_hh = Vil.get_KL_divergence_Samples(self.mu_weight_hh, Vil.softplus(self.rho_weight_hh), self.weight_hh, self.prior)
        
        KL_loss_bih = 0
        KL_loss_bhh = 0
        if self.bias is not None:
            KL_loss_bih = Vil.get_KL_divergence_Samples(self.mu_bias_ih, Vil.softplus(self.rho_bias_ih), self.bias_ih,  self.prior)
            KL_loss_bhh = Vil.get_KL_divergence_Samples(self.mu_bias_hh, Vil.softplus(self.rho_bias_hh), self.bias_hh,  self.prior)        
        KL_loss = KL_loss_ih + KL_loss_hh + KL_loss_bih +KL_loss_bhh
        return KL_loss
    
    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self._backend.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
    """
    Flag to set that we actually get the posterior mean and not a sample from the random variables
    """
    def set_posterior_mean(self, posterior_mean):
        self.posterior_mean = posterior_mean