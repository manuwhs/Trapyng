
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import Variational_inferences_lib as Vil
import pyTorch_utils as pytut
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

# changed configuration to this instead of argparse for easier interaction
CUDA = False
SEED = 1
BATCH_SIZE = 10
LOG_INTERVAL = 10
EPOCHS = 10

# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
ZDIMS = 20

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}


class VAE_reg1(nn.Module):
    def __init__(self, conf_a):
        
        self.LSTM_mode = 0
        super(VAE_reg1, self).__init__()
        
        self.loss_func = conf_a.loss_func
        self.cf_a = conf_a
        
        ## Use the linear model NN given by pyTorch that already does all the initialization
        ## and everything

        
        ############ ENCODER  #################
        
        ## 2 linearly dense layers !!
        if(self.LSTM_mode == 1):
            self.encLinear1  = PytorchSeq2VecWrapper(torch.nn.LSTM(1, hidden_size = conf_a.H_enc1, 
                                                   batch_first=True, bidirectional = False,
                                                   num_layers = 1, dropout = 0.0))
        else:
            self.encLinear1 = nn.Linear(conf_a.D_in, conf_a.H_enc1)

#        self.encLinear2 = nn.Linear(conf_a.H_enc1, conf_a.H_enc2)
        
        ######## Obtain the parameters of the latent space as a function of the reduced space ########
        self.encMu = nn.Linear(conf_a.H_enc1, conf_a.Z_dim)
        self.encRho = nn.Linear(conf_a.H_enc1, conf_a.Z_dim)

        
        self.activation_func_enc1 = conf_a.activation_func_enc1
#        self.activation_func_enc2 = conf_a.activation_func_enc2
        

        # DECODER
        self.decLinear1 = nn.Linear(conf_a.Z_dim, conf_a.H_dec1)
        self.decLinear2 = nn.Linear(conf_a.H_dec1, conf_a.D_in)
        
        self.activation_func_dec1 = conf_a.activation_func_dec1
        self.activation_func_dec2 = conf_a.activation_func_dec2

        ###### OPTIMIZER ########3
        optimizer = pytut.get_optimizers(self, self.cf_a)
        self._optimizer = optimizer
        
    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected
        21, fully connected 22)

        Parameters
        ----------
        x : [Nbatch, Dim_input] matrix;  or [Nbatch, Dim_input] 

        Returns
        -------

        (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS
            variance units one for each latent dimension

        """
        
        if(len(x.shape) == 1):
             x = x.view(1,x.shape[0])
            
        if (self.LSTM_mode == 1):
            x = x.view(x.shape[0],x.shape[1], 1)
            h_enc_1 = self.activation_func_enc1(self.encLinear1(x, None))  # type: Variable
        else:
            h_enc_1 = self.activation_func_enc1(self.encLinear1(x))  # type: Variable
            
        mu =  self.encMu(h_enc_1)
        rho =  self.encRho(h_enc_1)
        return mu, rho

    def decode(self, z: Variable) -> Variable:
        h_dec_1 = self.activation_func_dec1(self.decLinear1(z))
        o_estimation = self.decLinear2(h_dec_1)
        return o_estimation

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, rho = self.encode(x)
        z = self.sample_from_latent_space(mu, rho)
        o_estimation = self.decode(z)
        return o_estimation, mu, rho
    
    def sample_from_latent_space(self, mu: Variable, rho: Variable) -> Variable:
        """
           Just sample from the latent space. 
           We dont need to "reparametrice that much, only for the propagation of info
           but we are not training the mu and Sigma as parameters themselves as we
           would do in the VB algo.
        """

        if self.training:
            std = Vil.softplus(rho) # type: Variable
            return Vil.sample_gaussian(mu,std)
        else:
            return mu

    """
    ############### GET THE  LOSS ############
    """

    def get_KL_divergence(self, X = None, mu = None, std = None):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        """
        if (type(mu) == type(None)): # Compute the parts if we dont have them
            predictions, mu, rho = self.forward(X)
            std = Vil.softplus(rho)
        batch_size = X.shape[0]
        KL_loss = Vil.get_KL_divergence_hidden_space_VAE(mu, std)/batch_size

        return KL_loss
    
    def combine_losses(self, data_loss, KL_divergence, batch_size):
        KL_constant = self.cf_a.eta_KL /float((self.cf_a.Nsamples_train));
#        print (batch_size)
#        print ("KL constant: ", KL_constant)
        return  data_loss + KL_constant * KL_divergence


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
            
    def get_data_loss(self, X):
        """
        The loss of the data.
        TODO: Should I not create the graph here ?
        """
        X = X.to(device =self.cf_a.device )
#        print ("Size of Y", Y.shape, " Y ", Y)
#        print ("Size of predictions", self.forward(X).shape)
        with torch.no_grad():
            predictions, mu, rho = self.forward(X)
            return self.loss_func(predictions,X)

    def get_KL_loss(self,X):
        """
        Computes the KL div but without creating a graph !!
        """
        with torch.no_grad():
            KL_div = self.get_KL_divergence(X);
        return KL_div
    
    def get_loss(self, X):
        """ 
        Data Loss + VB loss
        """
        X = X.to(device =self.cf_a.device )
        with torch.no_grad():
            # Now we can just compute both losses which will build the dynamic graph
            data_loss = self.get_data_loss(X)
            KL_div = self.get_KL_divergence(X)
            batch_size = X.shape[0]
            total_loss =  self.combine_losses(data_loss, KL_div,batch_size)
        
        return total_loss

    """
      ############# TRAINING #################
    """
    def train_batch(self, X_batch):
        """
        It is enough to just compute the total loss because the normal weights 
        do not depend on the KL Divergence
        """
        # Now we can just compute both losses which will build the dynamic graph
        predictions, mu, rho = self.forward(X_batch)
        batch_size = mu.shape[0]
        data_loss = self.loss_func(predictions,X_batch)
        KL_div = self.get_KL_divergence(X_batch, mu = mu, std = Vil.softplus(rho))
        total_loss =  self.combine_losses(data_loss, KL_div, batch_size)
    
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
        
        