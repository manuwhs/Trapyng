from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.chdir("../../")
import import_folders

import graph_lib as gl
import numpy as np
## Import standard libraries
import tensorflow as tf

# Import libraries
import subprocess
import reader
import utilities_lib as utils 
## Import TF creared 
import BBB_LSTM_configs as Bconf
import BBB_LSTM_Model as BM
import Variational_inferences_lib as VI
import util_autoparallel as util

import pickle_lib as pkl

from graph_lib import gl
import numpy as np
import matplotlib.pyplot as plt

save_path = "../TensorBoard/BBB_LSTM/"

"""

Set the hyperparameters that we want
"""
mixing_pi = 0.25
prior_log_sigma1 = -1.0
prior_log_sigma2 = -7.0

"""
Set the data that we want !!
"""

############################################################################
############## Load the DATA #############################################
###########################################################################

"""
We will load the data into the RAM but we will until we build the graph
to transform it into TensorFlow elements and divide into batches with XXXX

"""
data_to_use = "aritificial"  #  ptb  aritificial

if (data_to_use == "ptb" ):
    model_select= "small"  # test small
    data_path = "../data"
    
    # Read the words from 3 documents and convert them to ids with a vocabulary
    raw_data = reader.ptb_raw_data(data_path)
    """
    Raw data contains 3 lists of a lot of words:
        - [0]: List of ids of the words for train
        - [1]: List of ids of the words for validation
        - [3]: List of ids of the words for validation
        - [4]: Number of words in the vocabulary.
    """
    
    train_data, valid_data, test_data, word_to_id, _ = raw_data
    # Create dictonary from ids to words.
    id_to_word = np.array(list(word_to_id.keys()))
    print (["Most common words: ", id_to_word[0:5]])

    # Create the objects with the hyperparameters that will be fed to the network
    train_config = Bconf.get_config(model_select,mixing_pi,prior_log_sigma1,prior_log_sigma2 )
    eval_config = Bconf.get_config(model_select,mixing_pi,prior_log_sigma1,prior_log_sigma2 )
    
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    
elif (data_to_use == "aritificial" ):
    model_select= "aritificial"  # test small
    Ndivisions = 10;
    folder_data = "../data/artificial/"
    
    X_list = pkl.load_pickle(folder_data +"X_values.pkl",Ndivisions)
    Y_list = pkl.load_pickle(folder_data +"Y_values.pkl",Ndivisions)
    
    num_steps, X_dim = X_list[0].shape
    num_chains = len(X_list)
    
    
    ## Divide in train val and test
    proportion_tr = 0.8
    proportion_val = 0.1
    proportion_tst = 1 -( proportion_val + proportion_tr)
    
    num_tr = 10000
    num_val = 5000
    num_tst = 5000
    
    train_X = [X_list[i] for i in range(num_tr)]
    train_Y = [Y_list[i] for i in range(num_tr)]
    
    val_X = [X_list[i] for i in range(num_tr, num_tr + num_val)]
    val_Y = [Y_list[i] for i in range(num_tr, num_tr + num_val)]
    
    tst_X = [X_list[i] for i in range(num_tr + num_val,  num_tr + num_val + num_tst)]
    tst_Y = [Y_list[i] for i in range(num_tr + num_val,  num_tr + num_val + num_tst)]

    # Create the objects with the hyperparameters that will be fed to the network
    train_config = Bconf.get_config(model_select,mixing_pi,prior_log_sigma1,prior_log_sigma2 )
    eval_config = Bconf.get_config(model_select,mixing_pi,prior_log_sigma1,prior_log_sigma2 )
    
    ###### Over Set parameters #####
    train_config.X_dim  = X_dim
    eval_config.X_dim  = X_dim
    train_config.num_steps  = num_steps
    eval_config.num_steps  = num_steps
    
    train_config.embedding  = False
    eval_config.embedding  = False
    
    train_config.Y_cardinality = 2
    eval_config.Y_cardinality= 2
    eval_config.batch_size = 20
    
############# PATH WHERE TO SAVE AND LATER LOAD THE OPERATIONS AND PARAMETERS AND RESULTS #############

## TensorBoard for seeing the intermediate results !
#subprocess.Popen(["tensorboard","--logdir="+ save_path])


save_path += "nl:"+str(train_config.num_layers) +"_nh:" + str(train_config.hidden_size) 
save_path += "_nt:" + str(train_config.num_steps) + "_Dx:" + str(train_config.X_dim) + "/"
utils.create_folder_if_needed(save_path)

#####################################################################################
#### TF graph for just creating the models. We create 3 models, one for train, validation

tf.reset_default_graph()

print ("BUILDING FIRST GRAPH")
#def change_random_seed(seed):
#    global prng
#    prng = np.random.RandomState(seed)
#    tf.set_random_seed(seed)
#    change_random_seed(global_random_seed)
build_Graph = 0
train_models = 0
test_models = 0
plot_data = 1
if (build_Graph):
    with tf.Graph().as_default():
        ## We need a global initializer for all the variables
        # Probably not used in the end ?
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                    train_config.init_scale, dtype = VI.data_type())
        """
        Now we are gonna create 3 instances of the Architecture. One for train, validation and test.
        The reason for this is...
        They will share the trained parameters, in this case the 
        
        The training architecture is special:
            - It has the is_training=True, so that the network is trained and XXX
            - In tensorBoard we will store a summary of the Losses so that later we can plot it and mange it
        
        SO WE SETTING THE DATA HERE INSTEAD OF PLACE HOLDERS ?
        """
            
        with tf.name_scope("Train"):
            """ Train input is the place_holder for the input data for a batch is placed
                - It is made of 2 tensors (like place holders):
                    TODO: Are placeholders like tensors with unknown dimensions that you can set it later ?
                    The X:  and Y.
            
            """
            # Train input contains the:
            #    - The hyperparameters which define the model
            #    - The data 
    
            if (data_to_use == "ptb" ):
                train_input = BM.BBB_LSTM_PTB_Input(batch_size = train_config.batch_size, num_steps = train_config.num_steps, data=train_data, name="TrainInput")
            elif (data_to_use == "aritificial" ):
                train_input = BM.BBB_LSTM_Artificial_Data_Input(batch_size = train_config.batch_size, 
                                                                X = train_X, Y = train_Y,  name="TrainInput")
        
           
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = BM.BBB_LSTM_Model(is_training=True, config=train_config, input_=train_input)
            
            # Variables to be used by stored at every call by tensorboard.
            tf.summary.scalar("Training_Loss", m.cost)
            tf.summary.scalar("Learning_Rate", m.lr)
            tf.summary.scalar("KL_Loss", m.kl_loss)
            tf.summary.scalar("Total_Loss", m.total_loss)
    
        """
        Now we define the validation and test Models. They are the same as the training mdoel.
        We set reuse=True so that it uses the same variables as the model Train ?
        And set is_training=False so that we do not do as many things in the code.
        
        Notice main differences are also:
            - To validation we give the valid_input.
            -
            - To test we give the test input. 
        """
        with tf.name_scope("Valid"):
    
            if (data_to_use == "ptb" ):
                valid_input = BM.BBB_LSTM_PTB_Input(batch_size = eval_config.batch_size, num_steps = eval_config.num_steps, data=valid_data, name="ValidInput")
            
            elif (data_to_use == "aritificial" ):
                valid_input = BM.BBB_LSTM_Artificial_Data_Input(batch_size = eval_config.batch_size, 
                                                                X = val_X, Y = val_Y,  name="ValidInput")
                
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = BM.BBB_LSTM_Model(is_training=False, config=eval_config, input_=valid_input)
            tf.summary.scalar("Validation_Loss", mvalid.cost)
    
        with tf.name_scope("Test"):
    
            if (data_to_use == "ptb" ):
                test_input = BM.BBB_LSTM_PTB_Input(batch_size = eval_config.batch_size, num_steps = eval_config.num_steps, data=test_data, name="ValidInput")
            
            elif (data_to_use == "aritificial" ):
                test_input = BM.BBB_LSTM_Artificial_Data_Input(batch_size = eval_config.batch_size, 
                                                                X = tst_X, Y = tst_Y,  name="TestInput")
                
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = BM.BBB_LSTM_Model(is_training=False, config=eval_config,
                                 input_=test_input)
    
        
        """
        Now we strore the models in the "Models" dictionary
        Then we export the operations 
        """
        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        
        metagraph = tf.train.export_meta_graph()
        
            ## If we have more GPUS allow for better optimization.
        soft_placement = False
        if train_config.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)


print ("BUILDING SECOND GRAPH")

if (train_models):
    with tf.Graph().as_default():
        
        """
        First we import the metagraph and operation that we just created.
        Why do we have 2 Graphs ? I dont know. 
        
        """
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        
        print ("Metagraph loaded")
        """
        Set the Supervisor and configuration
    
        """        
        
        print ("soft placements:", soft_placement)
        # TODO: Is the supervisor the one saving the state every X time ?
        sv = tf.train.Supervisor(logdir=save_path, save_model_secs = 10)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        print ("Supervisor and ConfigProto set")
        """
        Finally start a session with the Architecture. 
        We know have the Graph created,
            - Now we just have to fetch data from it (like the KL_loss)
              and in the train Model it will read the part where it will at the end update
              the values by gradiend Descent. It the other 2 it will not
            - We also dont need to fee it with data since it already has it in the config file somehow.
            
        In the session, for each of the epochs we will:
            - 
        """
        with sv.managed_session(config=config_proto) as session:
            print ("Session Initiated")
            for i in range(train_config.max_max_epoch):
                # Set the learning rate, setting in the internal variable of the model
                # It decays to 0 with the epochs !
                lr_decay = train_config.lr_decay ** max(i + 1 - train_config.max_epoch, 0.0)
                m.assign_lr(session, train_config.learning_rate * lr_decay)
                
                print ("########################## Epoch %i/%i (max) #########################" %(i+1, train_config.max_max_epoch))
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                
                train_perplexity = BM.run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = BM.run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
            test_perplexity = BM.run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
    
            if save_path:
                print("Saving model to %s." % save_path)
                sv.saver.save(session, save_path, global_step=sv.global_step)

if (test_models):
    ## Testing
    print ("Testing")
    predicted = []   # Variable to store predictions
    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            
           # session = tf.Session()
        
            test_perplexity = BM.run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
    
            print ("----------------------------------------------------------------")
            print ("------------------ Prediction of Output ---------------------")
    
           #  inputs, predicted = fetch_output(session, mtest)
    
            costs = 0.0
            state = session.run(model.initial_state)
    
            inputs = []
            outputs = []
            targets = []
            fetches = {
                "final_state": model.final_state,
                "output": model.output,
                "input": model.input_data,
                "targets": model.targets
            }
    
            for step in range(model.input.epoch_size):
                feed_dict = {}
                for i, (c, h) in enumerate(model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
    
                print ("Computing batch %i/%i"%(step, model.input.epoch_size))
                vals = session.run(fetches, feed_dict)
                state = vals["final_state"]
                output = vals["output"]
                input_i = vals["input"]
                
                outputs.append(output)
                inputs.append(input_i)
                targets.append(vals["targes"])
                if (step == 100):
                    break;

if (plot_data):
    
    batch_i = 25
    data = np.array(inputs[batch_i][0])[:,[0]]
    labels = np.array(targets[batch_i][0])[:]
    predicted = np.array(outputs[batch_i][0])[:,[1]]
    #print(data)
    #print(labels)
    #print (predicted)
    
    labels_chart = ["Example output for medium noise level", "","X[n]"]
    gl.set_subplots(3,1)
    ax1 = gl.plot(np.array(range(data.size)), data, nf = 1, labels = labels_chart, legend = ["X[n]"])
    ax2 = gl.stem(np.array(range(data.size)),labels, nf = 1, sharex = ax1, labels = ["","","Y[n]"], bottom = 0.5,
                  legend = ["Targets Y[n]"])
    gl.stem(np.array(range(data.size)),predicted, nf = 1, sharex = ax1, sharey = ax2, labels = ["","n","O[n]"],bottom = 0.5,
            legend = ["Predictions O[n]"])

    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)