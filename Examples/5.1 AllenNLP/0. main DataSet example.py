
# This cell just makes sure the library paths are correct. 
# You need to run this cell before you run the rest of this
# tutorial, but you can ignore the contents!
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
"""
Tokenization and Indexing
"""
from allennlp.data import Token
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data import Instance

"""
Vocabulary and Batch (padding and to Tensor of indexes)
"""
from allennlp.data import Vocabulary 
from allennlp.data.dataset import Batch

"""
Embedding and Encoding
"""
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


# Access the original strings and labels using the methods on the Fields.

def convert_to_instance(list_words,sentiment_label,PoS_labels):
    tokenized_words =  list(map(Token,list_words ))
    word_and_character_text_field = TextField( tokenized_words, 
                 token_indexers= {"tokens": SingleIdTokenIndexer(namespace="token_ids"), 
                                  "chars": TokenCharactersIndexer(namespace="token_chars")})
    sentiment_analysis_label_field = LabelField(sentiment_label, label_namespace="sentiment_tags")
    PoS_labels_field = SequenceLabelField(labels=PoS_labels,  label_namespace = "PoS_tags",
                                          sequence_field=word_and_character_text_field)
    
    instance_i = Instance({"text_field": word_and_character_text_field,
                           "label_sentiment": sentiment_analysis_label_field,
                           "Pos_labels": PoS_labels_field})
    return instance_i

"""
Create the text field for the Tokens.
For this purposes we tokenize the words of the document and then we create 2 token indexers
to have tokens both at word level and character level.
"""
## Format of the original data
list_words = ["The", "last", "song", "of", "Eminem", "was", "bad"]
sentiment_label = "negative"
PoS_labels = ["DET","ADV","NN", "DET","NN","VEB","ADJ"]

tokenized_words =  list(map(Token,list_words ))
word_and_character_text_field = TextField( tokenized_words, 
             token_indexers= {"tokens": SingleIdTokenIndexer(namespace="token_ids"), 
                              "chars": TokenCharactersIndexer(namespace="token_chars")})
print("Tokens in TextField: ", word_and_character_text_field.tokens)
"""
Now we create the labels for the sample. In sentiment analysis for example we could have.
Different problems have different type of labels:
     - Sequence to squence: A label per each of the tokens, like in next word prediction or PoS.
     - Sequence to label: A label for the chain, like in sentiment analysis.

"""
# In the case of a label for the entire instance then we use LabelField()
sentiment_analysis_label_field = LabelField(sentiment_label, label_namespace="sentiment_tags")
PoS_labels_field = SequenceLabelField(labels = PoS_labels, label_namespace = "PoS_tags",
                                      sequence_field=word_and_character_text_field)

print("Label of LabelField", sentiment_analysis_label_field.label)
print("Label of LabelField", PoS_labels_field.labels)

"""
Now we can create the Instance with the labels and the 
"""

instance1 = Instance({"text_field": word_and_character_text_field,
                       "label_sentiment": sentiment_analysis_label_field,
                        "Pos_labels": PoS_labels_field})
    
print("Fields in instance: ", instance1.fields)
"""
Lets create another instance to make a small dataset
"""
instance2 = convert_to_instance(["This", "is", "dope"], "positive", ["DET","VEB", "NN"])
instances = [instance1, instance2]

"""
########## CREATE A VOCABULARY ###################
We can automatically or by instances.
There is a vocabulary for each namespace (Labels and token indexers) !. 
In this case we have 2 for the input and 2 for the labels.
The vocabularies are Dict of Index -> Word and they include two special indicators. 
   - The padding
   - The unknown
This can work both for Chars and Words Tokens.

"""
# This will automatically create a vocab from our dataset.
# It will have "namespaces" which correspond to two things:
# 1. Namespaces passed to fields (e.g. the "tags" namespace we passed to our LabelField)
# 2. The keys of the 'Token Indexer' dictionary in 'TextFields'.
# passed to Fields (so it will have a 'tags' namespace).
vocab = Vocabulary.from_instances(instances)
print("This is the id -> word mapping for the 'token_ids' namespace: ")
print(vocab.get_index_to_token_vocabulary("token_ids"), "\n")
print("This is the id -> chars mapping for the 'token_chars' namespace: ")
print(vocab.get_index_to_token_vocabulary("token_chars"), "\n")

print("This is the id -> word mapping for the 'sentiment_tags' namespace: ")
print(vocab.get_index_to_token_vocabulary("sentiment_tags"), "\n")
print("This is the id -> word mapping for the 'PoS_tags' namespace: ")
print(vocab.get_index_to_token_vocabulary("PoS_tags"), "\n")

print("Vocab Token to Index dictionary: ", vocab._token_to_index, "\n")

### TODO: Create a Vocab from files or previously created or set of Tokens directly.

"""
################### CREATE A BATCH WITH THE INSTANCES ################
Next, we index our dataset using the generated vocabulary.
You must perform this step before  trying to generate arrays. 
Once the batch is created from the instances we can get:
    - The padded tokens, both in char level and word level
They are converted in arrays and are therefore ready for learning
AllenNLP works in batch_first mode !!
"""

batch = Batch(instances)
batch.index_instances(vocab)

# Get the padding lengths of the input namespaces. 
#TODO: It looks like it does show all of the namespaces ? 
# It does not show for the Sentiment LAabels.
padding_lengths = batch.get_padding_lengths()
print("Lengths used for padding: ", padding_lengths, "\n")

## Now we can get everything as a tensor to be used broh !!
tensor_dict = batch.as_tensor_dict(padding_lengths)
print(tensor_dict)

tokens_words_tensor = tensor_dict["text_field"]["tokens"]
tokens_chars_tensor = tensor_dict["text_field"]["chars"]
label_sentiment_tensor = tensor_dict["label_sentiment"]
label_PoS_tensor = tensor_dict["Pos_labels"]

print ("tokens_words_tensor shape: ",tokens_words_tensor.shape )
print ("tokens_chars_tensor shape: ",tokens_chars_tensor.shape )
print ("label_sentiment_tensor shape: ",label_sentiment_tensor.shape )
print ("label_PoS_tensor shape: ",label_PoS_tensor.shape )
"""
################ EMBEDDING AND ENCODING ###############
So far we just have each Chain sample in the Batch represented as 
a sequence of indexes (in all 4 categories (by words, charecters, 2 types of labels))

We want to Embed these indexes into vectors or real numbers that can be digested
by our learning algorithms. 
"""
Word_embedding_dim = 10
char_embeddedng_dim = 5
CNN_encoder_dim = 8
CNN_num_filters = 2

# The word embedding will transform every word to a "Word_embedding_dim" real valued vector
# Having a tensor (batch_size, max_sentence_length, Word_embedding_dim)
word_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_ids"), embedding_dim=Word_embedding_dim)

# The char embedding will transform every character into a ""char_embeddedng_dim" real valued vector
# Having a tensor (batch_size, max_sentence_length, max_word_length, char_embeddedng_dim)
char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_chars"), embedding_dim=char_embeddedng_dim)
# The Encoder will apply the cNN over the max_word_length dimension
# Having a tensor (batch_size, max_sentence_length, num_filters * ngram_filter_sizes)
character_cnn = CnnEncoder(embedding_dim=char_embeddedng_dim, num_filters=CNN_num_filters, output_dim=CNN_encoder_dim)

# We concatenate the char embdding and Encoding
token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, encoder=character_cnn)

### Now we finally create the finally embedder indicating what are the token ids it embedds
text_field_embedder = BasicTextFieldEmbedder(
        {"tokens": word_embedding, 
         "chars": token_character_encoder})

## Apply the Embedding to the batch 
    # This will have shape: (batch_size, sentence_length, word_embedding_dim + character_cnn_output_dim)
embedded_text = text_field_embedder(tensor_dict["text_field"])

print (embedded_text.shape)
dimensions = list(embedded_text.size())
print("Post embedding with our TextFieldEmbedder: ")
print("Batch Size: ", dimensions[0])
print("Sentence Length: ", dimensions[1])
print("Embedding Size: ", dimensions[2])