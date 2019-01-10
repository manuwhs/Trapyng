
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

"""
Library with the models
"""
import os
os.chdir("../../")
import import_folders

import PoS_tagging_utils as PoSut

torch.manual_seed(1)
"""
############# Hyperparameters ##############
"""
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

"""
#################### INITIALIZE THE DATA ########################
"""
reader = PoSut.PosDatasetReader()
train_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/training.txt'))
validation_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/validation.txt'))

"""
Create the vocabulary using the training and validation datasets.
"""
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)


"""
############### Token embeddings #################
Once we have the vocabulary of tokens
"""

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

"""
The LSTM is define here for some reason !! We could have just instantiated it in the 
model directly.
"""
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

"""
############### Instantiate the model and optimizer ##################
"""
model = PoSut.LstmTagger(word_embeddings, lstm, vocab)

optimizer = optim.SGD(model.parameters(), lr=0.1)

"""
############ Iterator that will get the samples for the problem #############
"""

iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

"""
##############################################################################
######################### TRAINING #######################################
Probably should not use this one because we want more features for the Bayesian elements.
This trainer should also save the model ? 
"""

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()

"""
############## Use the trained model ######################
We use an already implemented predictor that takes the model and how to preprocess the data
"""
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

tag_logits = predictor.predict("Eat my motherfucking jeans")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])