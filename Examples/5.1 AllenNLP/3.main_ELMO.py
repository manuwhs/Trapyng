
"""
ELMO !!
"""

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer


from allennlp.common.util import lazy_groups_of, prepare_global_logging
"""

"""
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch

from allennlp.data.tokenizers import  Tokenizer, WordTokenizer

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


"""
######################## Hyperparameters ##########################
"""

load_ELMO_flag = 0
"""
Loading pretrained ELMO
"""

if (load_ELMO_flag):
    print ("Loading ELMO")
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    print ("ELMO weights loaded")
    
"""
Encode a few sentences 
"""

# use batch_to_ids to convert sentences to character ids
sentences = ['First sentence pene.',
             'Another.', 
             "Third sentece that I am writing 7"]

## Create the indexers:
"""
############# CONVERT THE SEN
"""
tokenizer = WordTokenizer()
indexer = ELMoTokenCharactersIndexer()

def get_ELMO_text_field(sentence, indexer, tokenizer):
    """
    This function gets the ELMO text Field 
    """
    
    if (0):
        list_words = sentence.split(" ")
        ## Tokenize the words
        tokens = [Token(token) for token in list_words]
    else:
        tokens = tokenizer.tokenize(sentence)
    
    print ("Tokens of Sentence: ", tokens)
    
    # Create the ELMO field with the indexer
    field = TextField(tokens,{'character_ids': indexer})
    
    return field
    # Create the instance with the ELMO field
    
    
instances = []
for sentence in sentences:
    ## We tokenize every word. 
    field = get_ELMO_text_field(sentence, indexer, tokenizer)
    instance = Instance({"elmo": field})
    print("Fields in instance: ", instance.fields)
    instances.append(instance)


### Create a batch of the instances 
dataset = Batch(instances)

## Create an empty vocabulary ! We do not need to create one from dataset,
# It will use all of the indexer !!
vocab = Vocabulary()

## Create the index_instances from the batch, this will be used later by ELMO
dataset.index_instances(vocab)

"""
IMPORTANT: The ELMO uses just a character vocab in the interface.
It will compute the rest internally!

The ELMO words are padded to length 50 !
"""

character_ids = dataset.as_tensor_dict()['elmo']['character_ids']
print ("Shape of characters ids: ", character_ids.shape)
print ("Batch size: ", character_ids.shape[0])
print ("Maximum num words in batch: ", character_ids.shape[1])
print ("Maximum word length in dictionary: ", character_ids.shape[2])
#character_ids = batch_to_ids(sentences)

"""
Compute the Embeddings from the 
"""
embeddings = elmo(character_ids)

layer_1_values = embeddings["elmo_representations"][0]
layer_2_values = embeddings["elmo_representations"][1]
print ("Layer 1 representations: ", layer_1_values.shape)
print ("Layer 2 representations: ", layer_2_values.shape)
print ("Batch size: ", layer_1_values.shape[0])
print ("Maximum num words in batch: ", layer_1_values.shape[1])
print ("Word representation dimensionality: ", layer_1_values.shape[2])

