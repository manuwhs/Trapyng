
import torch
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from typing import Iterator, List, Dict
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
#from allennlp.nn.util import cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary

"""
Embedding and Encoding
"""
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

import glob
import string
import unicodedata
import os 
import numpy as np
from GeneralVBModelRNN import GeneralVBModelRNN
from LinearVB import LinearVB

from RNN_LSTM_lib import RNN_LSTM

from allennlp.data.iterators import BucketIterator

class config_architecture():

    
    ## Data dependent:
    task_type = "classification"
    D_in = None     # Dimensionality of input features
    D_out = None   # Dimensionality of output features

    ## DEVICES
    dtype = None  # Variable types
    device = None
    
    ##### Architecture ######
    Word_embedding_dim = 30
    CNN_encoder_dim = 0
    
    char_embeddedng_dim = 57
    CNN_num_filters = 5
    
    LSTM_H = 50
    Bayesian_Linear = False # If we want the last layer to be Bayesian
    Bayesian_LSTM = False # If we want the last layer to be Bayesian
    num_LSTM_layers = 1
    
    ### Nonlinearity
    activation_func = torch.tanh #   torch.cos  torch.clamp tanh
    
    ### Training 
    loss_func = None
    Nepochs = 200       # Number of epochs
    batch_size = 50     # Batch size
    
    ## The optimizer could be anything 
    optimizer = None
    optimizer_params = None
    lr = 0.01
    
    # Outp
    dop = 0.0 # Dropout p 
    
    
class NameCountryDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    all_letters = string.ascii_letters + " .,;'"
    def findFiles(path): return glob.glob(path)
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(list_words):
        list_processed_words = []
        for s in list_words:
            list_processed_words.append(''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in NameCountryDatasetReader.all_letters))
        return list_processed_words
            
    def readLines(filename):
        """
        Reads the lines of a file and coverts them to ASCII
        """
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        
        def get_words_as_biggest_tokends(line):
            return line.split(" ")
        def get_chars_as_biggest_tokends(line):
            return [line[i] for i in range(len(line))]
        
        return [NameCountryDatasetReader.unicodeToAscii(get_chars_as_biggest_tokends(line)) for line in lines]
        
    def __init__(self, cf_a, lazy:bool = False) -> None:
        """
        Initialize the reader, in this case with lazyness as well.
        We can specfy here the token_indexers if needed 
        """
        super().__init__(lazy=False)
        ## Default Token Indexer
        
        token_indexers = dict()
        
        if (cf_a.Word_embedding_dim > 0):
           token_indexers["tokens"] = SingleIdTokenIndexer(namespace="token_ids")
            
        if (cf_a.CNN_encoder_dim > 0):
            token_indexers["chars"] = TokenCharactersIndexer(namespace="token_chars")

        self.token_indexers = token_indexers
        
    def generate_instance(self, word_list: List[str], tags: List[str] = None) -> Instance:
        tokenized_words =  list(map(Token,word_list ))
        word_and_character_text_field = TextField( tokenized_words, 
                     token_indexers= self.token_indexers)
        
        if (type(tags) != type(None)):
            tags_Field = LabelField(tags, label_namespace="tags_country")
            instance_i = Instance({"text_field": word_and_character_text_field,
                                   "tags_field": tags_Field})
        else:
            instance_i = Instance({"text_field": word_and_character_text_field})
            
        return instance_i

    def _read(self, file_path: str = "../data/names/*.txt") -> Iterator[Instance]:
        
        files_path = NameCountryDatasetReader.findFiles(file_path)
        print (files_path)
        all_instances = []
        for filename in files_path:
            category = os.path.splitext(os.path.basename(filename))[0]
            lines = NameCountryDatasetReader.readLines(filename)
#            print (lines)
            for line in lines:
#                all_instances.append(self.generate_instance(line, category))
                
                yield self.generate_instance(line, category)
#        return all_instances


class NameCountryModel(Model):
    ## KL functions
    sample_posterior = GeneralVBModelRNN.sample_posterior
    get_KL_divergence = GeneralVBModelRNN.get_KL_divergence
    set_posterior_mean = GeneralVBModelRNN.set_posterior_mean
    combine_losses = GeneralVBModelRNN.combine_losses
    ## Interface functions
    predict = GeneralVBModelRNN.predict
    predict_proba = GeneralVBModelRNN.predict_proba
    get_data_loss = GeneralVBModelRNN.get_data_loss

    get_KL_loss = GeneralVBModelRNN.get_KL_loss
    
    
    def get_loss(self, text_field: Dict[str, torch.Tensor],
                tags_field: torch.Tensor = None):
        
        with torch.no_grad():
            predictions = self.forward(text_field)["tag_logits"]
            tags_field = tags_field.to(device = self.cf_a.device)
    #        print ("Predictions dtype: ", predictions.dtype)
    #        print ("labels dtype: ", Y_batch)
            data_loss = self.loss_func(predictions,tags_field)
            
            KL_div = self.get_KL_divergence()
            total_loss =  self.combine_losses(data_loss, KL_div)

        return total_loss

    def get_data_loss(self, text_field: Dict[str, torch.Tensor],
                tags_field: torch.Tensor = None):
        with torch.no_grad():
            predictions = self.forward(text_field)["tag_logits"]
            tags_field = tags_field.to(device = self.cf_a.device)
            
            data_loss = self.loss_func(predictions,tags_field)
            
        return data_loss
        
    def get_embedder(self, vocab, Word_embedding_dim, char_embeddedng_dim, CNN_num_filters, CNN_encoder_dim):
        # The word embedding will transform every word to a "Word_embedding_dim" real valued vector
        # Having a tensor (batch_size, max_sentence_length, Word_embedding_dim)
        
        indexers_dict = dict();
        if (Word_embedding_dim  > 0):
            word_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_ids"), embedding_dim=Word_embedding_dim)
            
            word_embedding = word_embedding.to(device = self.cf_a.device, dtype = self.cf_a.dtype)
            indexers_dict["tokens"] = word_embedding
        if (CNN_encoder_dim > 0):
            # The char embedding will transform every character into a ""char_embeddedng_dim" real valued vector
            # Having a tensor (batch_size, max_sentence_length, max_word_length, char_embeddedng_dim)
            char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_chars"), embedding_dim=char_embeddedng_dim)
            # The Encoder will apply the cNN over the max_word_length dimension
            # Having a tensor (batch_size, max_sentence_length, num_filters * ngram_filter_sizes)
            character_cnn = CnnEncoder(ngram_filter_sizes = (1,1), embedding_dim=char_embeddedng_dim, num_filters=CNN_num_filters, output_dim=CNN_encoder_dim)
            
            # We concatenate the char embdding and Encoding
            token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, encoder=character_cnn)
            
            token_character_encoder = token_character_encoder.to(device = self.cf_a.device, dtype = self.cf_a.dtype)
            indexers_dict["chars"] = token_character_encoder
        ### Now we finally create the finally embedder indicating what are the token ids it embedds
        text_field_embedder = BasicTextFieldEmbedder(indexers_dict)
    
        return text_field_embedder
    
    def get_sec2vec_encoder(self, input_dim, output_dim):
        """
        LSTM from which we will pre
        
        """
        
        if (0):
            lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(input_dim, output_dim, batch_first=True))
        else:
            
            lstm = RNN_LSTM(input_size=input_dim, hidden_size = output_dim, bias = True, 
                            batch_first=True, num_layers = self.cf_a.num_LSTM_layers,
                            Bayesian = self.cf_a.Bayesian_LSTM, prior = self.cf_a.LSTM_prior)
        return lstm

    def __init__(self, cf_a, # Configuration file
                 vocab: Vocabulary) -> None:
        ## We send the vocabulary to the upper model.
        # Apparently AllenNLP needs the vocabulary
        super().__init__(vocab)
        
        self.cf_a = cf_a
        self.loss_func = cf_a.loss_func
        self.prior = cf_a.LSTM_prior
        """
        Token Embedding Biatch !! 
        """
        self.word_embeddings = self.get_embedder(vocab, cf_a.Word_embedding_dim, 
                                            cf_a.char_embeddedng_dim, cf_a.CNN_num_filters, cf_a.CNN_encoder_dim)
        
        
        self.encoder = self.get_sec2vec_encoder(cf_a.CNN_encoder_dim +  cf_a.Word_embedding_dim, cf_a.LSTM_H)
        
        
        if (cf_a.Bayesian_Linear):
            self.hidden2tag = LinearVB(in_features=self.cf_a.LSTM_H,
                                              out_features=vocab.get_vocab_size('tags_country'), bias = True, 
                                              prior = cf_a.Linear_output_prior)
        else:
            self.hidden2tag = torch.nn.Linear(in_features=self.cf_a.LSTM_H,
                                              out_features=vocab.get_vocab_size('tags_country'))
        
        self.accuracy = CategoricalAccuracy()
        
        """
        List of Bayesian Linear Models.
        Using this list we can easily set the special requirements of VB models.
        And also analize easily the weights in the network
        """
        self.VBmodels = []
        self.LinearModels = []
        
        if (cf_a.Bayesian_Linear):
            self.VBmodels.append(self.hidden2tag)
        else:
            self.LinearModels.append(self.hidden2tag)
        
        if (cf_a.Bayesian_LSTM):
            self.VBmodels.extend(self.encoder.get_LSTMCells())
        else:
            self.LinearModels.extend(self.encoder.get_LSTMCells())
            
    def forward(self,
                text_field: Dict[str, torch.Tensor],
                tags_field: torch.Tensor = None) -> torch.Tensor:
    
        """
        Put the data into CUDA if that is the device.
        No need to change the type, it is indices long
        """
        
        if (self.cf_a.CNN_encoder_dim > 0):
            text_field["chars"] = text_field["chars"].to(device = self.cf_a.device)
    
        if (self.cf_a.Word_embedding_dim > 0):
            text_field["tokens"] = text_field["tokens"].to(device = self.cf_a.device)
        
        """
        The input is a the ??
        """
        self.sample_posterior()
        

        ## TODO: What is the mask ? For the final output.
        verbose = 0
        seq2seq_flag = False # Most likely does not make sense in seq2vec, it is used in the end for the loggits and shit.
        
        if(seq2seq_flag):
            mask = get_text_field_mask(text_field)
        else:
            mask = None
        
        if(verbose):
            print ("------------------ New Batch ---------------")
            if (self.cf_a.CNN_encoder_dim > 0):
                print ("Text_field_chars (",text_field["chars"].shape,"): \n" , text_field["chars"])
        
            if (self.cf_a.Word_embedding_dim > 0):
                print ("Text_field_words (",text_field["tokens"].shape,"):\n" , text_field["tokens"])
        
        if(seq2seq_flag):
            print ("Mask: (",mask.shape,"):\n" , mask)
        ## Propagate the data !
        embeddings = self.word_embeddings(text_field)
        if(verbose):
            print ("Embeddings size: ", embeddings.shape)
        
        encoder_out = self.encoder(embeddings, mask)[-1][0]
        
        tag_logits = self.hidden2tag(encoder_out)
        if(verbose):
            print ("tags_field (", tags_field.shape, ") : \n", tags_field)
            print ("tag_logits (", tag_logits.shape, ") : \n", tag_logits)
        ## We return the output as a dictionary
        output = {"tag_logits": tag_logits}
        
        if tags_field is not None:
            tags_field = tags_field.to(device = self.cf_a.device)
            
            self.accuracy(tag_logits, tags_field, mask)
            if (seq2seq_flag):
                output["loss"] = sequence_cross_entropy_with_logits(tag_logits, tags_field, mask)
            else:
                 output["loss"] = self.cf_a.loss_func(tag_logits, tags_field) 
        return output # ["tag_logits"]

    def train_batch(self, X_batch, Y_batch):
        """
        It is enough to just compute the total loss because the normal weights 
        do not depend on the KL Divergence
        """
        # Now we can just compute both losses which will build the dynamic graph
        predictions = self.forward(X_batch)["tag_logits"]
        Y_batch = Y_batch.to(device = self.cf_a.device)
        
#        print ("Predictions dtype: ", predictions.dtype)
#        print ("labels dtype: ", Y_batch)

        
        data_loss = self.loss_func(predictions,Y_batch)
        
        KL_div = self.get_KL_divergence()
        total_loss =  self.combine_losses(data_loss, KL_div)

        
        self.zero_grad()     # zeroes the gradient buffers of all parameters
        total_loss.backward()
        
        if (type(self.cf_a.optimizer) == type(None)):
            parameters = filter(lambda p: p.requires_grad, self.parameters())
            with torch.no_grad():
                for f in parameters:
                    f.data.sub_(f.grad.data * self.lr )
        else:
#            print ("Training")
            self.cf_a.optimizer.step()
            self.cf_a.optimizer.zero_grad()
        return total_loss
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    
    
    def get_confusion_matrix(self,dataset, batch_size = 100 ):
        """
        This function will get the confusion matrix of all the data given
        by creating the iterator.
        """
        with torch.no_grad():
            classes_vocab = self.vocab.get_index_to_token_vocabulary("tags_country")
            classes = []
            for i in range(2,len(classes_vocab)):
                classes.append(classes_vocab[i])
                
            Nclasses = len(classes)
            confusion = np.zeros((Nclasses,Nclasses))
            
            ## Compute all the predictions using the batches !!
            iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("text_field", "num_tokens")])
            iterator.index_with(self.vocab)
            num_batches = int(np.floor(len(dataset)/batch_size))
          
            	# Create the iterator over the data:
            batches_iterable = iterator(dataset)
        
            # Compute the validation accuracy by using all the Validation dataset but in batches.
    #        predictions = []
            for j in range(num_batches):
                batch = next(batches_iterable)
                tensor_dict = batch # Already converted
                tag_logits = self(tensor_dict["text_field"])['tag_logits'].detach().cpu().numpy()
                tag_ids = list(np.argmax(tag_logits, axis=-1))
    #            predictions.extend(tag_ids)
                Y = tensor_dict["tags_field"].detach().cpu().numpy().flatten().tolist()
                
#                print (tag_ids)
#                print (Y)
                # We substract 2 due to unknown and padding
                for i in range(len(tag_ids)):  # For each prediction
                    confusion[Y[i]-2, tag_ids[i]-2] += 1
                
            for i in range(Nclasses):
                confusion[i] = confusion[i] / confusion[i].sum()
    
        return classes, confusion
    
    
        
        
        