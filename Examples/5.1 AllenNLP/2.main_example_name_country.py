"""
  Thoughts on DataReader:
           To allow lazy mode we should add the lazy flag in the initialization and the _read() function should not return a list of instances but
           rather yield them within a loop.
           When calling read() of the DataReader, if it is not lazy mode it will read the Iterator (Yielder) until it has all samples and will join them into a list and return them.
           If we have lazy mode then we only call read() once as well, and then every time we iterate over the return element, it will fetch the data from _read() and return the instance.
           instances = reader.read('my_instances.txt')
           for epoch in range(10):
               for instance in instances:
                   process(instance)

            Then each epoch's for instance in instances results in a new call to MyDatasetReader._read(), and your instances will be read from disk 10 times.

        AllenNLP uses a DataIterator abstraction to iterate over datasets using configurable batching, shuffling, and so on.
        (the padding is taken care of by the Batching, but we could order the sequences in our dataset do have batches with same lenghts, this is what iterators can do).

        The included dataset readers all work with lazy datasets, but they have extra options that you might want to specify in that case.

       How do you know when the yield iterator is over ? because it returns None ? for the non lazy loading it is fine, we have full control.

        _max_instances_in_memory _instances_per_epoch

         BucketIterator, which can order your instances by e.g. sentence length.

      ---
      In the seq2seq entropy loss we used the mask as the vector of weights for the elements in the chains.

      --
      For using the Iterator, once the iterator object is initialized, we just call __call__ with the iterable from the data reader and it will
      output another iterable with the batches, already converted to tensors of indexes using the vocabulary.
      We could also use the function: get_num_batches(self, instances: Iterable[Instance]) but how is this going to work for lazy mode ? do we have to load everything once to test it ? 
      Ok, for lazy mode it wount work unless we specify the number of samples in each epoch. for non lazy it will but it also will ahve do reload everything, so not worth it.
          Shold we check at each iteration if the batch result is None and the call the iterator again ? 

"""
import torch
import torch.optim as optim
import numpy as np

from allennlp.common.file_utils import cached_path

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.data.dataset import Batch
"""
Library with the models
"""
import os
os.chdir("../../")
import import_folders

import PoS_tagging_utils as PoSut
import Name_country_utils as Ncut
import pyTorch_utils as pytut
import Variational_inferences_lib as Vil

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import plotting_func as pf
# Specific utilities
import utilities_lib as ul

"""
######################## OPTIONS ##############################
"""
folder_images = "../pics/Pytorch/BasicExamples_Bayesian_RNN_class_gpu_fullVB/"
video_fotograms_folder  = folder_images +"video_Bayesian1/"
video_fotograms_folder2  = folder_images +"video_Bayesian2/"
video_fotograms_folder3 = folder_images +"video_Bayesian3/"
video_fotograms_folder4 = folder_images +"video_Bayesian4/"
folder_model = "../models/pyTorch/Basic0_Bayesian_RNN/"

################ PLOTTING OPTIONS ################
create_video_training = 1
Step_video = 1

"""
###################### Setting up func ###########################
"""
## Windows and folder management
plt.close("all") # Close all previous Windows
ul.create_folder_if_needed(folder_images)
ul.create_folder_if_needed(video_fotograms_folder)
ul.create_folder_if_needed(video_fotograms_folder2)
ul.create_folder_if_needed(video_fotograms_folder3)
ul.create_folder_if_needed(video_fotograms_folder4)
ul.create_folder_if_needed(folder_model)

if(1):
    ul.remove_files(video_fotograms_folder)
    ul.remove_files(video_fotograms_folder2)
    ul.remove_files(video_fotograms_folder3)
    ul.remove_files(video_fotograms_folder4)
    
## Set up seeds for more reproducible results
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

"""
############# Hyperparameters ##############
"""

dtype = torch.float
device = pytut.get_device_name(cuda_index = 0)
criterion = torch.nn.CrossEntropyLoss()

## LOAD THE BASE DATA CONFIG
cf_a = Ncut.config_architecture

## Set data and device parameters
cf_a.dtype = dtype  # Variable types
cf_a.device = device

# Set other training parameters
cf_a.loss_func = criterion

"""
######################## Bayesian Prior ! ########################
"""
log_sigma_p1 = np.log(0.1)
log_sigma_p2 = np.log(0.3)
pi = 0.5
prior = Vil.Prior(pi, log_sigma_p1,log_sigma_p2)
cf_a.Linear_output_prior = prior
cf_a.LSTM_prior = prior

"""
#################### INITIALIZE THE DATA ########################
"""
reader = Ncut.NameCountryDatasetReader(cf_a)
train_dataset = reader.read('../data/RNN_text/names/*.txt')
validation_dataset = reader.read('../data/RNN_text/names/*.txt')
"""
Create the vocabulary using the training and validation datasets.
"""

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

print("This is the id -> word mapping for the 'token_ids' namespace: ")
print(vocab.get_index_to_token_vocabulary("token_ids"), "\n")
print("This is the id -> chars mapping for the 'token_chars' namespace: ")
print(vocab.get_index_to_token_vocabulary("token_chars"), "\n")
print("This is the id -> word mapping for the 'tags_country' namespace: ")
print(vocab.get_index_to_token_vocabulary("tags_country"), "\n")

"""
############### Instantiate the model and optimizer ##################
"""

model = Ncut.NameCountryModel(cf_a, vocab)
optimizer = optim.SGD(model.parameters(), lr=0.01)
cf_a.optimizer = optimizer

model.to(device = device, dtype = dtype)
"""
############ Iterator that will get the samples for the problem #############
"""
batch_size=10
batch_size_validation = 100

iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("text_field", "num_tokens")])
iterator.index_with(vocab)

iterator_validation = BucketIterator(batch_size = batch_size_validation, sorting_keys=[("text_field", "num_tokens")])
iterator_validation.index_with(vocab)

num_batches = int(np.floor(len(train_dataset)/batch_size))
num_batches_validation = int(np.floor(len(validation_dataset)/batch_size_validation))
	# Create the iterator over the data:
batches_iterable = iterator(train_dataset)
batches_iterable_validation = iterator_validation(validation_dataset)

"""
##############################################################################
######################### TRAINING #######################################
Probably should not use this one because we want more features for the Bayesian elements.
This trainer should also save the model ? 
"""

def get_training_values (model, vocab, train_dataset, validation_dataset,
                         tr_data_loss, val_data_loss, KL_loss,final_loss_tr, final_loss_val, batch_size=100):
    model.eval()
    model.set_posterior_mean(True)

    data_loss_validation = 0
    data_loss_train = 0
    loss_validation = 0
    loss_train = 0
    
    # Create own iterators for this:
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("text_field", "num_tokens")])
    iterator.index_with(vocab)
    
    iterator_validation = BucketIterator(batch_size = batch_size, sorting_keys=[("text_field", "num_tokens")])
    iterator_validation.index_with(vocab)
    
    num_batches = int(np.floor(len(train_dataset)/batch_size))
    num_batches_validation = int(np.floor(len(validation_dataset)/batch_size_validation))
    	# Create the iterator over the data:
    batches_iterable = iterator(train_dataset)
    batches_iterable_validation = iterator(validation_dataset)

    # Compute the validation accuracy by using all the Validation dataset but in batches.
    for j in range(num_batches_validation):
        batch = next(batches_iterable_validation)
        tensor_dict = batch # Already converted
        data_loss_validation += model.get_data_loss(tensor_dict["text_field"],tensor_dict["tags_field"])
        loss_validation += model.get_loss(tensor_dict["text_field"],tensor_dict["tags_field"])
 
    data_loss_validation = data_loss_validation/num_batches_validation
    loss_validation = loss_validation/num_batches_validation
    
    ## Same for training
    for j in range(num_batches):
        batch = next(batches_iterable)
        tensor_dict = batch # Already converted
        data_loss_train += model.get_data_loss(tensor_dict["text_field"],tensor_dict["tags_field"])
        loss_train += model.get_loss(tensor_dict["text_field"],tensor_dict["tags_field"])
    
    data_loss_train = data_loss_train/num_batches
    loss_train = loss_train/num_batches
    
    tr_data_loss.append(data_loss_train)
    val_data_loss.append(data_loss_validation)
    KL_loss.append(-model.get_KL_loss())
    final_loss_tr.append(loss_train)
    final_loss_val.append(loss_validation)

    model.train()
    model.set_posterior_mean(False)
    
    
# We could also know the real samples corresponding to the training right ? index_batch ? or somethign like this, to be checked.
num_epochs = 100

tr_data_loss = []
val_data_loss = [] 

KL_loss = []
final_loss_tr = []
final_loss_val = []

for i in range(num_epochs):

    if (i == 0):
        get_training_values (model,vocab, train_dataset, validation_dataset,
                             tr_data_loss, val_data_loss, KL_loss,final_loss_tr, final_loss_val, batch_size=100)
        print ("Initial Losses: Train(%.2f) Val(%.2f) KL(%.2f)"%(tr_data_loss[-1], val_data_loss[-1], KL_loss[-1]))
        if (create_video_training):
            pf.create_image_weights_epoch(model, video_fotograms_folder2, i)
            pf.create_Bayesian_analysis_charts_simplified(model, train_dataset, validation_dataset,
                                                tr_data_loss, val_data_loss, KL_loss,
                                                video_fotograms_folder4, i)
    print ("Doing epoch: %i"%(i+1))
    for j in range(num_batches):
        batch = next(batches_iterable)
        tensor_dict = batch # Already converted
        model.train_batch(tensor_dict["text_field"],tensor_dict["tags_field"])
#            print ("Batch %i/%i"%(j,num_batchs ))
        
    get_training_values (model,vocab, train_dataset, validation_dataset,
                         tr_data_loss, val_data_loss, KL_loss,final_loss_tr, final_loss_val, batch_size=100)
    print ("Losses: Train(%.2f) Val(%.2f) KL(%.2f)"%(tr_data_loss[-1], val_data_loss[-1], KL_loss[-1]))
    if (create_video_training):
        pf.create_image_weights_epoch(model, video_fotograms_folder2, i)
        pf.create_Bayesian_analysis_charts_simplified(model ,train_dataset, validation_dataset,
                                            tr_data_loss, val_data_loss, KL_loss,
                                            video_fotograms_folder4, i+1)

#            output = model(tensor_dict["text_field"],tensor_dict["tags_field"])
#            loss = output["loss"] # We can get the loss coz we gave the labels as input



			# gradient and everything. 
"""
############## Use the trained model ######################
We use an already implemented predictor that takes the model and how to preprocess the data
"""

name_exmaple = "Eat my motherfucking jeans"
name_exmaple = "Carlos Sanchez"
tokens_list = [name_exmaple[i] for i in range(len(name_exmaple))]
Instance_test = reader.generate_instance(tokens_list,None)
batch = Batch([Instance_test])
batch.index_instances(vocab)

padding_lengths = batch.get_padding_lengths()
tensor_dict = batch.as_tensor_dict(padding_lengths)

model.eval()
tag_logits = model(tensor_dict["text_field"])['tag_logits'].detach().cpu().numpy()
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'tags_country') for i in tag_ids])