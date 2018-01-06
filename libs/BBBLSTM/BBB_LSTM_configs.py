
from tensorflow.python.client import device_lib

class ArtificialDataConfig(object):
    """Small config."""
    
    ############ Architecture size hyperparameters  ############
    num_layers = 2    # Number of layers of LSTMs of the network
    hidden_size = 23  # Dimensionality of the hidden space of the LTSM
                      # This is also the num of hidden neurons in each of the
                      # 4 gates of the LSTM layers.
                      
    num_steps = 50    # Number of time unrollings applied to the chains.
                      # That is the length of the chains, which is kept contant
                      # instead of be able to accept chains of different length
                      # TODO: Make a network that accepts chains of different length
                      # Easy for test if we rebuild the model for a specific chain I guess.
                      # Probably way harder to program and less efficient in parallel computation
                      # Also the problem of weighting the Loss of chains with different length.
                      
                      # The chains presented to the network must in principle contain this length.
                      # They should be chopped otherwise to have this number of elements.
                      
    ########### Training Hyperparameters ##################
    max_max_epoch = 1    # Number of maximum epochs for training 
    max_epoch = 4        # No idea.

    batch_size = 3    # Number of chains in a batch. We can compute the number of
                       # batches in which a big sequence of data can be divided into by
                       # dividing it by [num_steps x batch_size]


    init_scale = 0.1      # TODO No idea
    learning_rate = 1.0   # Initial learning rate
    lr_decay = 0.8        # Exponential decay of the learning rate
    
    # Probably to do with the regularization of Maremba
    max_grad_norm = 5
    keep_prob = 1.0
 
    ########### Parameters of the input ##################
    """
    The input X can be continuous or discrete:
        - If continuous: Then the size of the input vectors is the size of the
          original vectors and that is it. We do not need to specify anything.
        - If discrete: Then we need to embedd them into random vectors. The dimensionality
          of this vectors will be given in the variable "size"
          
    The targets Y are discrete values:
        This is a classifier system with noiseless tags Y.
        We need to specify the cardinality of Y in order to know how many 
        output neurons we need in the outtermost softmax layer.
        
    """
    # If the inputs are words and we want to embedd them with a dictionary,
    # this is the size of the dictionary and therefore the size of the input
    Y_cardinality = 2   # Cardinality of the output !!
    X_dim = None      # Dimensionality of the embedding of the categorical variables 
                      # into random vectors of dimension "size"
    embedding = True
    
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 15    
    max_epoch = 4
    max_max_epoch = 5
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 10
    
    Y_cardinality = 10000   # Cardinality of the output !!
    X_dim = 30      # Dimensionality of the embedding of the categorical variables 

    embedding = True
    
class MediumConfig(object):
    """
    Medium config.
    Slightly modified according to email.
    """
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 20
    max_max_epoch = 70
    keep_prob = 1.0
    lr_decay = 0.9
    batch_size = 20
    vocab_size = 10000

    embedding = True
class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000

    embedding = True
class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

    embedding = True
    
def get_config(model_type, prior_pi, log_sigma1, log_sigma2):
    """Get model config."""
    print ("Using Model configuration: %s"%model_type)
    if model_type == "small":
        config = SmallConfig()
    elif model_type == "medium":
        config = MediumConfig()
    elif model_type == "large":
        config = LargeConfig()
    elif model_type == "test":
        config = TestConfig()
    elif model_type == "aritificial":
        config = ArtificialDataConfig()
    else:
        raise ValueError("Invalid model: %s", model_type)

    config.prior_pi = prior_pi
    config.log_sigma1 = log_sigma1
    config.log_sigma2 = log_sigma2

    ########### Automatically get the number of GPUs we have ##################
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
#    print(len(gpus))
    if len(gpus) == 0:
        config.num_gpus = 1
        # TODO: We need to set it to at least one.
    else:
        config.num_gpus = len(gpus)
        print ("$$$$$$$$$$$ YOU ACTUALLY HAVE GPUs DUDE $$$$$$$$$$$")
    return config