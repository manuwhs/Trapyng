
import os
import numpy as np
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


class Generator(nn.Module):
    def __init__(self, conf_a):
        super(Generator, self).__init__()
        self.cf_a = conf_a
        self.LSTM_mode = 0
        
        if(self.LSTM_mode == 0):
            self.encLinear1 = nn.Linear(conf_a.Z_dim, conf_a.H_enc1)
        else:
            self.encLinear1  = PytorchSeq2VecWrapper(torch.nn.LSTM(conf_a.Z_dim, hidden_size = conf_a.H_enc1, 
                                                   batch_first=True, bidirectional = False,
                                                   num_layers = 1, dropout = 0.0))
        # GENERATOR
        self.activation_func_enc1 = conf_a.activation_func_enc1
        self.hidden_to_signal = nn.Linear(conf_a.H_enc1, conf_a.D_in)

        ## Optimizer
        self.optimizer = pytut.get_optimizers(self, self.cf_a)
        
    def forward(self, z: Variable) -> (Variable):
        """
        Take a noise vector and convert it into a sample of the dataset.

        Parameters
        ----------
        z : [Nbatch, Dim_noise] matrix;

        Returns
        -------

        x : [Nbatch, Dim_input] matrix

        """
        # Convert the noise to a hidden space
        
        if(len(z.shape) == 1):
             z = z.view(self.cf_a.Z_dim,z.shape[0])
            
        if (self.LSTM_mode == 1):
            z = z.view(z.shape[0],z.shape[1], self.cf_a.Z_dim)
            h_enc_1 = self.activation_func_enc1(self.encLinear1(z, None))  # type: Variable
        else:
            h_enc_1 = self.activation_func_enc1(self.encLinear1(z))  # type: Variable
            
        h_enc_1 = torch.tanh(self.encLinear1(z))
        
        # Convert the hidden space to the generated sample
        generated_x = self.hidden_to_signal(h_enc_1)
        
        return generated_x

    def get_input_sampler(self):
#        return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
        return lambda m, n: torch.Tensor(np.random.normal(0, 1, (m, n)))  # Gaussian
    
    def generate_samples(self, batch_size):
        caca = self.get_input_sampler()
        input_noise = Variable(caca(batch_size, self.cf_a.Z_dim))
        input_noise = input_noise.to(device = self.cf_a.device)

        generated_samples = self.forward(input_noise)  # detach to avoid training G on these labels
        generated_samples.to(device = self.cf_a.device)
        return generated_samples
    
class Discriminator(nn.Module):
    def __init__(self, conf_a):
        super(Discriminator, self).__init__()
        self.cf_a = conf_a
            
        self.loss = nn.BCELoss() 
        
        # DISCRIMNATOR
        self.discLinear1 = nn.Linear(conf_a.D_in, conf_a.H_dec1)
        self.discLinear2 = nn.Linear(conf_a.H_dec1, 1)
        
        ## Optimizer
        self.optimizer = pytut.get_optimizers(self, self.cf_a)
        
    def forward(self, x: Variable) -> Variable:
        """
        Take an example input signal (either generated or real) and using 
        a parametric model, convert into a single value from which we can
        compute if it is a real or fake sample.

        Parameters
        ----------
        x : [Nbatch, Dim_noise] matrix;

        Returns
        -------

        o : [Nbatch, 1] matrix

        """
        h_disc_1 = torch.tanh(self.discLinear1(x))
        o_estimation = torch.sigmoid(self.discLinear2(h_disc_1))
        
        return o_estimation
    
    def get_loss(self, x: Variable, t: Variable) -> Variable:
        """
        Get loss of the samples given
        """
        decision = self.forward(x)
#        print (decision.shape)
#        print (t.shape)
        loss = self.loss(decision,t.to(device =  x.device))  # ones = true

        return loss


class GAN_reg1(nn.Module):
    def __init__(self, conf_a):
        self.cf_a = conf_a
        super(GAN_reg1, self).__init__()


        ### Initialize Generator and Discriminator ###
        self.Gen = Generator(conf_a).to(device = conf_a.device)
        self.Disc = Discriminator(conf_a).to(device = conf_a.device)

    """
      ############# TRAINING #################
    """
    def train_Disc_batch(self, X_batch, generated_samples = None):
        # 1. Train D on real+fake
        self.Disc.zero_grad()
        #  1A: Train D on real
        D_loss_real = self.Disc.get_loss(X_batch, Variable(torch.ones([X_batch.shape[0],1])))  # ones = true
        
       #  1B: Train D on fake
        if type(generated_samples) == type(None):
            generated_samples = self.Gen.generate_samples(X_batch.shape[0]).detach()  # detach to avoid training G on these labels
        else:
            generated_samples = generated_samples.detach()
            
        D_loss_generated = self.Disc.get_loss(generated_samples, Variable(torch.zeros([X_batch.shape[0],1])))  # ones = true
        
        # Train generator
        
        Total_loss = (D_loss_real + D_loss_generated)/2
        Total_loss.backward();
        
#        D_loss_real.backward() # compute/store gradients, but don't change params
#        D_loss_generated.backward()
        self.Disc.optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

        return Total_loss
    
    def train_Gen_batch(self, X_batch):

        # 2. Train G on D's response (but DO NOT train D on these labels)
        self.Gen.zero_grad()

       #  1B: Train G on fake
        generated_samples = self.Gen.generate_samples(X_batch.shape[0])
        G_loss_generated = self.Disc.get_loss(generated_samples, Variable(torch.ones([X_batch.shape[0],1])))  # ones = true
        
        # Train Generator
        G_loss_generated.backward()
        self.Gen.optimizer.step()  # Only optimizes G's parameters

        return generated_samples, G_loss_generated
    
    def train_batch(self, X_batch):
        # 1. Train D on real+fake
        generated_samples, G_loss = self.train_Gen_batch(X_batch) 
        D_loss = self.train_Disc_batch(X_batch, generated_samples)

        return G_loss, D_loss
    
#        dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
    
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
        
        