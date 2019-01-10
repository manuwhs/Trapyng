"""
Operations to be used in the library
"""


## Import standard libraries
import tensorflow as tf
# Import libraries


def apply_nonlinearity(Z, name = "linear", params = None):
    """ 
    Function to easily select from nonlinearities
    """
    
    if (name == "tanh"):
        O = tf.nn.tanh(Z)       
        
    elif(name == "relu"):
        O = tf.nn.relu(Z)        
        
    elif(name == "linear"):
        O = Z    

    elif(name == "softmax"):
        O =  tf.nn.softmax(Z) 
        
    return O