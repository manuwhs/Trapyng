
"""
Initial file where we played arounds with the elemnts of the Bidaf architecture
before putting them in a model.
"""
            
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder

import torch
from allennlp.models import load_archive
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer


from allennlp.common.util import lazy_groups_of, prepare_global_logging
"""

"""
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch

from allennlp.data.tokenizers import  Tokenizer, WordTokenizer

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
"""
Embedding and Encoding
"""
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.modules.similarity_functions import LinearSimilarity

"""
Loses and accuracies
"""
from torch.nn.functional import nll_loss
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

"""
############ OWN LIBRARY 
"""
import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

from squad1_reader import Squad1Reader
from bidaf_model import BidirectionalAttentionFlow_1

"""
 FLAGS
"""
load_dataset_from_disk = 0  # Load the dataset from disk and comput the Tokens and Indexing

use_ELMO = 1             # Use the ELMO tokenization - idexing - encodding
load_ELMO_experiments_flag = 0  # To not having to load ELMO every time doing experiments

class config_bidaf():
    # Embedding parameters
    use_ELMO = True
    
    # Highway parameters
    num_highway_layers = 2
    
    ## Phrase layer parameters
    phrase_layer_dropout = 0.2
    phrase_layer_num_layers = 1
    phrase_layer_hidden_size = 100
    
    # Modelling Passage parameters
    modeling_passage_dropout = 0.2
    modeling_passage_num_layers = 2
    modeling_passage_hidden_size = 100
    
    # Span end encoding parameters
    span_end_encoder_dropout = 0.2
    modeling_passage_num_layers = 2
    modeling_passage_hidden_size = 100
    
    # Masking parameters
    mask_lstms = True
    
    # spans output parameters
    spans_output_dropout  = 0.2
    
"""
################ LOAD PRETRAINED MODEL ###############
We load the pretrained model and we can see what parts it has and maybe reused if needed.
"""

if (load_dataset_from_disk):
    train_squad1_file = "../data/squad/train-v1.1.json"
    
    tokenizer = None
    token_indexers = None
    if (use_ELMO):
        tokenizer = WordTokenizer()
        token_indexers = {'character_ids': ELMoTokenCharactersIndexer()} 
    
    ## Create the Data Reader with the Tokenization and indexing
    squad_reader = Squad1Reader(tokenizer = tokenizer, token_indexers = token_indexers)
    train_dataset = squad_reader.read(file_path = train_squad1_file)

    """
    The same context can have different quesitons, eachs of them being an invidual question.
    The 
    """
    print ("Number of training instances: %i"% len(train_dataset))
    print ("Instances fields: ",train_dataset[0].fields.keys() )
    print ("Passage: ",train_dataset[0].fields["passage"] )
    print ("Question: ",train_dataset[0].fields["question"] )
    print ("span_start: ",train_dataset[0].fields["span_start"] )
    print ("span_end: ",train_dataset[0].fields["span_end"] )
    print ("metadata: ",train_dataset[0].fields["metadata"].metadata )

"""
############ CREATE EMPTY VOCABULARY AND SAMPLE BATCH ###############
Create a sample Batch to propagate through the network 
"""
# We do not need vocabulary in this case, all possible chars ?
vocab = Vocabulary()
use_custom_example_flag = 0
if (use_custom_example_flag):
    ### Create a batch of the instances 
    passage_text = "One time I was writing a unit test, and it succeeded on the first attempt."
    question_text = "What kind of test succeeded on its first attempt?"
    char_spans = [(6, 10)]
    instance = squad_reader.text_to_instance(question_text, 
                                               passage_text, 
                                               char_spans = char_spans)
    
    print ("Keys instance: ", instance.fields.keys())
    
    # Batch intances and convert to index using the vocabulary.
    instances = [instance]
else:

    instances = [train_dataset[0],train_dataset[1]]

## Create the batch ready to be used
dataset = Batch(instances)
dataset.index_instances(vocab)

print ("-------------- DATASET EXAMPLE ---------------")
character_ids_passage = dataset.as_tensor_dict()['passage']['character_ids']
character_ids_question = dataset.as_tensor_dict()['question']['character_ids']

question =  dataset.as_tensor_dict()['question']
passage =  dataset.as_tensor_dict()['passage']
span_start =  dataset.as_tensor_dict()['span_start']
span_end =  dataset.as_tensor_dict()['span_end']
metadata =  dataset.as_tensor_dict()['metadata']

print ("Shape of characters ids passage: ", character_ids_passage.shape)
print ("Shape of characters ids question: ", character_ids_question.shape)

print ("Batch size: ", character_ids_passage.shape[0])
print ("Maximum num words in batch: ", character_ids_passage.shape[1])
print ("Maximum word length in dictionary: ", character_ids_passage.shape[2])

"""
---------------- OBTAINING MASK -------------------------
"""
print ("-------------- MASK BATCH ---------------")
mask_lstms = True
question_mask = util.get_text_field_mask(question).float()
passage_mask = util.get_text_field_mask(passage).float()
question_lstm_mask = question_mask if mask_lstms else None
passage_lstm_mask = passage_mask if mask_lstms else None

print ("question mask dimensions: ", question_mask.shape)
#print (question_mask)
print ("passage mask dimensions: ", passage_mask.shape)
#print (passage_mask)

#question_lstm_mask = None; passage_lstm_mask = None


"""
################### EMBEDDING LAYER  #########################################
"""
print ("-------------- EMBEDDING LAYER ---------------")
if (use_ELMO):
    if (load_ELMO_experiments_flag):
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        print ("Loading ELMO")
        text_field_embedder = Elmo(options_file, weight_file, 2, dropout=0)
        print ("ELMO weights loaded")
else:
    text_field_embedder = TextFieldEmbedder()
    token_embedders = dict()
    text_field_embedder = Embedding(embedding_dim = 100, trainable = False)

## Parameters needed for the next layer
embedder_out_dim = text_field_embedder.get_output_dim()

print ("Embedder output dimensions: ", embedder_out_dim)
## Propagate the Batch though the Embedder
embeddings_batch_question = text_field_embedder(character_ids_question)["elmo_representations"][1]
embeddings_batch_passage = text_field_embedder( character_ids_passage)["elmo_representations"][1]

#print (embeddings_batch_question)
print ("Question representations: ", embeddings_batch_question.shape)
print ("Passage representations: ", embeddings_batch_passage.shape)
print ("Batch size: ", embeddings_batch_question.shape[0])
print ("Maximum num words in question: ", embeddings_batch_question.shape[1])
print ("Word representation dimensionality: ", embeddings_batch_question.shape[2])

batch_size = embeddings_batch_question.size(0)
passage_length = embeddings_batch_passage.size(1)

"""
################### Highway LAYER  #########################################
The number of highway layers to use in between embedding the input and passing it through
the phrase layer.
"""
print ("-------------- HIGHWAY LAYER ---------------")

num_highway_layers = 2

highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))

highway_batch_question = highway_layer(embeddings_batch_question)
highway_batch_passage = highway_layer(embeddings_batch_passage)

print ("Question representations: ", highway_batch_question.shape)
print ("Passage representations: ", highway_batch_passage.shape)
print ("Maximum num words in question: ", highway_batch_question.shape[1])
print ("Word representation dimensionality: ", highway_batch_question.shape[2])

"""
################### phrase_layer LAYER  #########################################
NOTE: Since the LSTM implementation of PyTorch cannot apply dropout in the last layer, 
we just apply ourselves later
"""

print ("-------------- PHRASE LAYER ---------------")

dropout = 0.2
bidirectional = True

## Create dropout layer
if dropout > 0:
    dropout_phrase_layer = torch.nn.Dropout(p=dropout)
else:
    dropout_phrase_layer = lambda x: x

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedder_out_dim, hidden_size = 100, 
                                           batch_first=True, bidirectional = True,
                                           num_layers = 1, dropout = dropout))
phrase_layer = lstm # PytorchSeq2SeqWrapper(lstm)

## Propagate the question and answer:
encoded_question = dropout_phrase_layer(phrase_layer(highway_batch_question, question_lstm_mask))
encoded_passage = dropout_phrase_layer(phrase_layer(highway_batch_passage, passage_lstm_mask))

# The dimensionality of the co
encoding_dim = encoded_question.size(-1)
print ("encoding_dim: ", encoding_dim)
print ("Question encoding: ", encoded_question.shape)
print ("Passage encoding: ", encoded_passage.shape)
"""
################### SIMILARITY FUNCTION LAYER  #########################################
NOTE: Since the LSTM implementation of PyTorch cannot apply dropout in the last layer, 
we just apply ourselves later
"""

print ("-------------- SIMILARITY LAYER ---------------")

similarity_function = LinearSimilarity(
      combination = "x,y,x*y",
      tensor_1_dim =  200,
      tensor_2_dim = 200)

matrix_attention = LegacyMatrixAttention(similarity_function)

passage_question_similarity = matrix_attention(encoded_passage, encoded_question)
# Shape: (batch_size, passage_length, question_length)
print ("passage question similarity: ", passage_question_similarity.shape)


# Shape: (batch_size, passage_length, question_length)
passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
# Shape: (batch_size, passage_length, encoding_dim)
passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

# We replace masked values with something really negative here, so they don't affect the
# max below.
masked_similarity = util.replace_masked_values(passage_question_similarity,
                                               question_mask.unsqueeze(1),
                                               -1e7)
# Shape: (batch_size, passage_length)
question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
# Shape: (batch_size, passage_length)
question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
# Shape: (batch_size, encoding_dim)
question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
# Shape: (batch_size, passage_length, encoding_dim)
tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                            passage_length,
                                                                            encoding_dim)

# Shape: (batch_size, passage_length, encoding_dim * 4)
final_merged_passage = torch.cat([encoded_passage,
                                  passage_question_vectors,
                                  encoded_passage * passage_question_vectors,
                                  encoded_passage * tiled_question_passage_vector],
                                 dim=-1)

"""
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
"""
print ("-------------- Modelling LAYER ---------------")

dropout = 0.2
bidirectional = True

## Create dropout layer
if dropout > 0:
    dropout_modeled_passage = torch.nn.Dropout(p=dropout)
else:
    dropout_modeled_passage = lambda x: x

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(encoding_dim * 4, hidden_size = 100, 
                                           batch_first=True, bidirectional = True,
                                           num_layers = 2, dropout = dropout))
modeling_layer = lstm 

modeled_passage = dropout_modeled_passage(modeling_layer(final_merged_passage, passage_lstm_mask))
modeling_dim = modeled_passage.size(-1)

print ("Modeled passage shape: ", modeled_passage.shape)


print ("-------------- SPAN START REPRESENTATION ---------------")
encoding_dim = phrase_layer.get_output_dim()
modeling_dim = modeling_layer.get_output_dim()
span_start_input_dim = encoding_dim * 4 + modeling_dim
span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))
        
 # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
span_start_input = dropout_modeled_passage(torch.cat([final_merged_passage, modeled_passage], dim=-1))
# Shape: (batch_size, passage_length)
span_start_logits = span_start_predictor(span_start_input).squeeze(-1)
# Shape: (batch_size, passage_length)
span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

# Shape: (batch_size, modeling_dim)
span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
# Shape: (batch_size, passage_length, modeling_dim)
tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                           passage_length,
                                                                           modeling_dim)

print ("-------------- SPAN END REPRESENTATION ---------------")

dropout = 0.2
bidirectional = True

## Create dropout layer
if dropout > 0:
    dropout_span_end = torch.nn.Dropout(p=dropout)
else:
    dropout_span_end = lambda x: x

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(encoding_dim * 4 + modeling_dim * 3, hidden_size = 100, 
                                           batch_first=True, bidirectional = True,
                                           num_layers = 2, dropout = dropout))
span_end_encoder = lstm 

span_end_encoding_dim = span_end_encoder.get_output_dim()
span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

# Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
span_end_representation = torch.cat([final_merged_passage,
                                     modeled_passage,
                                     tiled_start_representation,
                                     modeled_passage * tiled_start_representation],
                                    dim=-1)
# Shape: (batch_size, passage_length, encoding_dim)
encoded_span_end = dropout_span_end(span_end_encoder(span_end_representation,
                                                        passage_lstm_mask))
# Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
span_end_input = dropout_span_end(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
span_end_logits = span_end_predictor(span_end_input).squeeze(-1)
span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

print ("-------------- LOGITS OF BOTH SPANS and BEST SPAN ---------------")

span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)

best_span = BidirectionalAttentionFlow_1.get_best_span(span_start_logits, span_end_logits)

print ("best_spans", best_span)

"""
------------------------------ GET LOSES AND ACCURACIES -----------------------------------
"""
span_start_accuracy_function = CategoricalAccuracy()
span_end_accuracy_function = CategoricalAccuracy()
span_accuracy_function = BooleanAccuracy()
squad_metrics_function = SquadEmAndF1()

# Compute the loss for training.
if span_start is not None:
    span_start_loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
    span_end_loss = nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
    loss = span_start_loss + span_end_loss
    
    span_start_accuracy_function(span_start_logits, span_start.squeeze(-1))
    span_end_accuracy_function(span_end_logits, span_end.squeeze(-1))
    span_accuracy_function(best_span, torch.stack([span_start, span_end], -1))

    span_start_accuracy = span_start_accuracy_function.get_metric()
    span_end_accuracy =  span_end_accuracy_function.get_metric()
    span_accuracy = span_accuracy_function.get_metric()


    print ("Loss: ", loss)
    print ("span_start_accuracy: ", span_start_accuracy)
    print ("span_start_accuracy: ", span_start_accuracy)
    print ("span_end_accuracy: ", span_end_accuracy)
    
# Compute the EM and F1 on SQuAD and add the tokenized input to the output.
if metadata is not None:
    best_span_str = []
    question_tokens = []
    passage_tokens = []
    for i in range(batch_size):
        question_tokens.append(metadata[i]['question_tokens'])
        passage_tokens.append(metadata[i]['passage_tokens'])
        passage_str = metadata[i]['original_passage']
        offsets = metadata[i]['token_offsets']
        predicted_span = tuple(best_span[i].detach().cpu().numpy())
        start_offset = offsets[predicted_span[0]][0]
        end_offset = offsets[predicted_span[1]][1]
        best_span_string = passage_str[start_offset:end_offset]
        best_span_str.append(best_span_string)
        answer_texts = metadata[i].get('answer_texts', [])
        if answer_texts:
            squad_metrics_function(best_span_string, answer_texts)
            squad_metrics = squad_metrics_function.get_metric()
    print ("Best spans str: ", best_span_str)
    print ("Squad accuracies: ", squad_metrics)
            
            
