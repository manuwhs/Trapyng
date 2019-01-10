import math

from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.nn import Activation, util
import Variational_inferences_lib as Vil
import numpy as np

class LinearSimilarityVB(SimilarityFunction):
    """
    This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.
    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.
    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.
    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : ``str``, optional (default="x,y")
        Described above.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                 activation: Activation = None,
                 prior = None) -> None:
        super(LinearSimilarityVB, self).__init__()
        self._combination = combination
        combined_dim = util.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        
        self.posterior_mean = False # Flag to know if we sample from the posterior mean or we actually sample
        
        ## If no prior is specified we just create it ourselves
        if (type(prior) == type (None)):
            prior = Vil.Prior(0.5, np.log(0.1),np.log(0.5))
        
        size_combination = int(torch.Tensor(combined_dim).size()[0])
#        print ("Combination size: ", size_combination)
        prior =  prior.get_standarized_Prior(size_combination)
        self.prior = prior 
        
        """
        Mean and rhos of the parameters
        """
        self.mu_weight = Parameter(torch.Tensor(combined_dim))# , requires_grad=True
        self.rho_weight = Parameter(torch.Tensor(combined_dim))

        self.rho_bias = Parameter(torch.Tensor(1))
        self.mu_bias = Parameter(torch.Tensor(1))
            
        """
        The sampled weights
        """
        self.weight = torch.Tensor(combined_dim)
        self.bias = torch.Tensor(1)
        
        self._activation = activation or Activation.by_name('linear')()
        
        ## Initialize the Variational variables
        self.reset_parameters()
#        self.sample_posterior()

    def reset_parameters(self):
#        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
#        self._weight_vector.data.uniform_(-std, std)
#        self._bias.data.fill_(0)
        
        self.rho_weight.data = Vil.init_rho(self.mu_weight.size(), self.prior)
        self.rho_bias.data = Vil.init_rho(self.mu_bias.size(), self.prior)
        
        ## Now initialize the mean
        self.mu_weight.data = Vil.init_mu(self.mu_weight.size(), 
                                          self.prior,Ninput = self.mu_weight.size(0), type = "LinearSimilarity")
        
        self.mu_bias.data = Vil.init_mu(self.mu_bias.size(), self.prior, Ninput = self.mu_weight.size(0), type = "LinearSimilarity")
        
    def sample_posterior(self):
        """
        This function samples the Bayesian weights from the parameters and puts them into the variables.
        It needs to do so using the reparametrization trick so that we can derive respect to sigma and mu
        """
        
#        print ("SAMPLING FROM LINEAR SIMILARITY VB")
        if (self.posterior_mean == False):
            self.weight = Vil.sample_posterior(self.mu_weight, Vil.softplus(self.rho_weight))
            self.bias = Vil.sample_posterior(self.mu_bias, Vil.softplus(self.rho_bias))
#            print (self.bias)
        else:
            self.weight.data = self.mu_weight.data
            self.bias.data = self.mu_bias.data
                
    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = util.combine_tensors(self._combination, [tensor_1, tensor_2])
        dot_product = torch.matmul(combined_tensors, self.weight)
        return self._activation(dot_product + self.bias)

    def get_KL_divergence(self):
        """
        This function computes the KL loss for all the Variational Weights in the network !!
        It does not sample the weights again, it uses the ones that are already sampled.
        
        """
        KL_loss_W = Vil.get_KL_divergence_Samples(self.mu_weight, Vil.softplus(self.rho_weight), self.weight, self.prior)
        KL_loss_b = 0
        if self.bias is not None:
            KL_loss_b = Vil.get_KL_divergence_Samples(self.mu_bias, Vil.softplus(self.rho_bias), self.bias,  self.prior)
            
        KL_loss = KL_loss_W + KL_loss_b
        
        return KL_loss
    
    """
    Flag to set that we actually get the posterior mean and not a sample from the random variables
    """
    def set_posterior_mean(self, posterior_mean):
        self.posterior_mean = posterior_mean
        