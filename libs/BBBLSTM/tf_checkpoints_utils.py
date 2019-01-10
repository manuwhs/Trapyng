"""
This library will contain a set of code to deal with the saving and restoring of the variables in a graph.
Ideally for a given model, we would like to store periodically
    - Trainable variables (Weights and other parameters)
    - Non trainable variables (global step, learning rate, loss function value, evaluation variables....)

What we want:
    During the training, checkpoints will be created with the values of these variables.
    If the training is stopped, we can resume it using these values.
    After the training process we can access the variables of each checkpoint to analyze the training. 

Steps:
    - Create a Saver that saves the checkpoints
    - Function to load the last checkpoint and set the variables. The graph needs to build before.
    
"""

## Import standard libraries
import tensorflow as tf
import os
from tensorflow.python.tools import inspect_checkpoint as chkp
import utilities_lib as ul

# Import libraries

def create_saver(self, path):
    """
    Create the appropiate Saver for the variables
    """
    
    # Note, if we don’t specify anything in the tf.train.Saver(), it saves all the variables
    saver = tf.train.Saver(max_to_keep=10000)
    
    if (0):  # Other options
        #saves a model every 2 hours and maximum 4 latest models are saved.
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    
    
    # Variable to store the number of iteration of train,  Very common practice in ternsorflow.
    # gloval_step will be tf variable modify by the optimizer everytime we call it. 
    
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    self.variables_path = path
    
    self.saver = saver
        
def save_variables(self):
    self.saver.save(self.session, self.variables_path + self.name, global_step = self.global_step)
    
    # (sess, 'my-model', global_step=step,write_meta_graph=False)
    
def get_all_checkpoints_paths(path):
    """
    This function gets all the checkpints in the folder, since each of them has a file like:
        General_model1-61.index
    """
    
    all_paths =  ul.get_allPaths(path)
    
    model_checkpoints = []
    for path in all_paths:
        name = path.split("/")[-1]
        if (name.find(".index") != -1):
            name = name.split(".index")[0]
            model_checkpoints.append(name)
    
    def get_number(element):
        return int(element.split("-")[-1])
    model_checkpoints.sort(key = get_number)
    return model_checkpoints

def get_all_checkpoints_gloabl_step(model_checkpoints):
    """
    This function returns the numbers of the global step for each cluster
    """ 
    return [int(x.split("-")[-1]) for x in model_checkpoints]


def restore_variables(self, checkpoint_name = None):
    """
    Restore the variables from the Saved ones.
    If no specific version is selected, it wil load the last checkpoint
    
    Returns:
        True if variables were loaded
        False: Otherwise
    """
    
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.variables_path))
    
    if ckpt:
        if (type(checkpoint_name) == type(None)):
#            print ("Restoring latest checkpoint from: " + str(ckpt.model_checkpoint_path))
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
#            print ("Restoring checkpoint from: " + str(self.variables_path + checkpoint_name))
            self.saver.restore(self.session, str(self.variables_path + checkpoint_name))
        # HACK : global step is not restored for some unknown reason
        last_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        # assign to global step
        self.session.run(self.global_step.assign(last_step))
        
        return True
    return False

def return_variables(self):
    chkp.print_tensors_in_checkpoint_file(self.variables_path  + self.name ,tensor_name='Weights', all_tensors=True)

    


        








