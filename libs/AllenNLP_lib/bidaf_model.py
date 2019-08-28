import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
import Variational_inferences_lib as Vil
from allennlp.common import squad_eval
import time
import sys 
from bidaf_utils import send_error_email
import numpy as np
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

"""
SEQUENTIAL MODELS
"""
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
"""
SIMILARITY
"""
from allennlp.modules.similarity_functions import LinearSimilarity
import gc
"""
OWN LIBRARY
"""
from GeneralVBModelRNN import GeneralVBModel
import pyTorch_utils as pytut
import bidaf_utils as bidut

# Bayesian models
from LinearVB import LinearVB
from LinearSimilarityVB import LinearSimilarityVB
from HighwayVB import HighwayVB
# @Model.register("bidaf1")


class BidirectionalAttentionFlow_1(Model):
    """
    This class implements a Bayesian version of Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).
    """
    
    def __init__(self, vocab: Vocabulary, cf_a, preloaded_elmo = None) -> None:
        super(BidirectionalAttentionFlow_1, self).__init__(vocab, cf_a.regularizer)
        
        """
        Initialize some data structures 
        """
        self.cf_a = cf_a
        # Bayesian data models
        self.VBmodels = []
        self.LinearModels = []
        """
        ############## TEXT FIELD EMBEDDER with ELMO ####################
        text_field_embedder : ``TextFieldEmbedder``
            Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
        """
        if (cf_a.use_ELMO):
            if (type(preloaded_elmo) != type(None)):
                text_field_embedder = preloaded_elmo
            else:
                text_field_embedder = bidut.download_Elmo(cf_a.ELMO_num_layers, cf_a.ELMO_droput )
                print ("ELMO loaded from disk or downloaded")
        else:
            text_field_embedder = None
        
#        embedder_out_dim  = text_field_embedder.get_output_dim()
        self._text_field_embedder = text_field_embedder
        
        if(cf_a.Add_Linear_projection_ELMO):
            if (self.cf_a.VB_Linear_projection_ELMO):
                prior = Vil.Prior(**(cf_a.VB_Linear_projection_ELMO_prior))
                print ("----------------- Bayesian Linear Projection ELMO --------------")
                linear_projection_ELMO =  LinearVB(text_field_embedder.get_output_dim(), 200, prior = prior)
                self.VBmodels.append(linear_projection_ELMO)
            else:
                linear_projection_ELMO = torch.nn.Linear(text_field_embedder.get_output_dim(), 200)
        
            self._linear_projection_ELMO = linear_projection_ELMO
            
        """
        ############## Highway layers ####################
        num_highway_layers : ``int``
            The number of highway layers to use in between embedding the input and passing it through
            the phrase layer.
        """
        
        Input_dimension_highway = None
        if (cf_a.Add_Linear_projection_ELMO):
            Input_dimension_highway = 200
        else:
            Input_dimension_highway = text_field_embedder.get_output_dim()
            
        num_highway_layers = cf_a.num_highway_layers
        # Linear later to compute the start 
        if (self.cf_a.VB_highway_layers):
            print ("----------------- Bayesian Highway network  --------------")
            prior = Vil.Prior(**(cf_a.VB_highway_layers_prior))
            highway_layer = HighwayVB(Input_dimension_highway,
                                                          num_highway_layers, prior = prior)
            self.VBmodels.append(highway_layer)
        else:
            
            highway_layer = Highway(Input_dimension_highway,
                                                          num_highway_layers)
        highway_layer = TimeDistributed(highway_layer)
        
        self._highway_layer = highway_layer
        
        """
        ############## Phrase layer ####################
        phrase_layer : ``Seq2SeqEncoder``
            The encoder (with its own internal stacking) that we will use in between embedding tokens
            and doing the bidirectional attention.
        """
        if cf_a.phrase_layer_dropout > 0:       ## Create dropout layer
            dropout_phrase_layer = torch.nn.Dropout(p=cf_a.phrase_layer_dropout)
        else:
            dropout_phrase_layer = lambda x: x
        
        phrase_layer = PytorchSeq2SeqWrapper(torch.nn.LSTM(Input_dimension_highway, hidden_size = cf_a.phrase_layer_hidden_size, 
                                                   batch_first=True, bidirectional = True,
                                                   num_layers = cf_a.phrase_layer_num_layers, dropout = cf_a.phrase_layer_dropout))
        
        phrase_encoding_out_dim = cf_a.phrase_layer_hidden_size * 2
        self._phrase_layer = phrase_layer
        self._dropout_phrase_layer = dropout_phrase_layer
        
        """
        ############## Matrix attention layer ####################
        similarity_function : ``SimilarityFunction``
            The similarity function that we will use when comparing encoded passage and question
            representations.
        """
        
        # Linear later to compute the start 
        if (self.cf_a.VB_similarity_function):
            prior = Vil.Prior(**(cf_a.VB_similarity_function_prior))
            print ("----------------- Bayesian Similarity matrix --------------")
            similarity_function = LinearSimilarityVB(
                  combination = "x,y,x*y",
                  tensor_1_dim =  phrase_encoding_out_dim,
                  tensor_2_dim = phrase_encoding_out_dim, prior = prior)
            self.VBmodels.append(similarity_function)
        else:
            similarity_function = LinearSimilarity(
                  combination = "x,y,x*y",
                  tensor_1_dim =  phrase_encoding_out_dim,
                  tensor_2_dim = phrase_encoding_out_dim)
            
        matrix_attention = LegacyMatrixAttention(similarity_function)
        self._matrix_attention = matrix_attention
        
        """
        ############## Modelling Layer ####################
        modeling_layer : ``Seq2SeqEncoder``
            The encoder (with its own internal stacking) that we will use in between the bidirectional
            attention and predicting span start and end.
        """
        ## Create dropout layer
        if cf_a.modeling_passage_dropout > 0:       ## Create dropout layer
            dropout_modeling_passage = torch.nn.Dropout(p=cf_a.modeling_passage_dropout)
        else:
            dropout_modeling_passage = lambda x: x
        
        modeling_layer = PytorchSeq2SeqWrapper(torch.nn.LSTM(phrase_encoding_out_dim * 4, hidden_size = cf_a.modeling_passage_hidden_size, 
                                                   batch_first=True, bidirectional = True,
                                                   num_layers = cf_a.modeling_passage_num_layers, dropout = cf_a.modeling_passage_dropout))

        self._modeling_layer = modeling_layer
        self._dropout_modeling_passage = dropout_modeling_passage
        
        """
        ############## Span Start Representation #####################
        span_end_encoder : ``Seq2SeqEncoder``
            The encoder that we will use to incorporate span start predictions into the passage state
            before predicting span end.
        """
        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        
        # Linear later to compute the start 
        if (self.cf_a.VB_span_start_predictor_linear):
            prior = Vil.Prior(**(cf_a.VB_span_start_predictor_linear_prior))
            print ("----------------- Bayesian Span Start Predictor--------------")
            span_start_predictor_linear =  LinearVB(span_start_input_dim, 1, prior = prior)
            self.VBmodels.append(span_start_predictor_linear)
        else:
            span_start_predictor_linear = torch.nn.Linear(span_start_input_dim, 1)
            
        self._span_start_predictor_linear = span_start_predictor_linear
        self._span_start_predictor = TimeDistributed(span_start_predictor_linear)

        """
        ############## Span End Representation #####################
        """
        
        ## Create dropout layer
        if cf_a.span_end_encoder_dropout > 0:
            dropout_span_end_encode = torch.nn.Dropout(p=cf_a.span_end_encoder_dropout)
        else:
            dropout_span_end_encode = lambda x: x
        
        span_end_encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(encoding_dim * 4 + modeling_dim * 3, hidden_size = cf_a.modeling_span_end_hidden_size, 
                                                   batch_first=True, bidirectional = True,
                                                   num_layers = cf_a.modeling_span_end_num_layers, dropout = cf_a.span_end_encoder_dropout))
   
        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        
        self._span_end_encoder = span_end_encoder
        self._dropout_span_end_encode = dropout_span_end_encode
        
        if (self.cf_a.VB_span_end_predictor_linear):
            print ("----------------- Bayesian Span End Predictor--------------")
            prior = Vil.Prior(**(cf_a.VB_span_end_predictor_linear_prior))
            span_end_predictor_linear = LinearVB(span_end_input_dim, 1, prior = prior)
            self.VBmodels.append(span_end_predictor_linear) 
        else:
            span_end_predictor_linear = torch.nn.Linear(span_end_input_dim, 1)
        
        self._span_end_predictor_linear = span_end_predictor_linear
        self._span_end_predictor = TimeDistributed(span_end_predictor_linear)

        """
        Dropput last layers
        """
        if cf_a.spans_output_dropout > 0:
            dropout_spans_output = torch.nn.Dropout(p=cf_a.span_end_encoder_dropout)
        else:
            dropout_spans_output = lambda x: x
        
        self._dropout_spans_output = dropout_spans_output
        
        """
        Checkings and accuracy
        """
        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(Input_dimension_highway , phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_end_encoder.get_input_dim(), 4 * encoding_dim + 3 * modeling_dim,
                               "span end encoder input dim", "4 * encoding dim + 3 * modeling dim")

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        """
        mask_lstms : ``bool``, optional (default=True)
            If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
            with only a slight performance decrease, if any.  We haven't experimented much with this
            yet, but have confirmed that we still get very similar performance with much faster
            training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
            required when using masking with pytorch LSTMs.
        """
        self._mask_lstms = cf_a.mask_lstms

    
        """
        ################### Initialize parameters ##############################
        """
        #### THEY ARE ALL INITIALIZED WHEN INSTANTING THE COMPONENTS ###
    
        """
        ####################### OPTIMIZER ################
        """
        optimizer = pytut.get_optimizers(self, cf_a)
        self._optimizer = optimizer
        #### TODO: Learning rate scheduler ####
        #scheduler = optim.ReduceLROnPlateau(optimizer, 'max')
    
    def forward_ensemble(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                get_sample_level_information = False) -> Dict[str, torch.Tensor]:
        """
        Sample 10 times and add them together
        """
        self.set_posterior_mean(True)
        most_likely_output = self.forward(question,passage,span_start,span_end,metadata,get_sample_level_information)
        self.set_posterior_mean(False)
       
        subresults = [most_likely_output]
        for i in range(10):
           subresults.append(self.forward(question,passage,span_start,span_end,metadata,get_sample_level_information))

        batch_size = len(subresults[0]["best_span"])

        best_span = bidut.merge_span_probs(subresults)
        
        output = {
                "best_span": best_span,
                "best_span_str": [],
                "models_output": subresults
        }
        if (get_sample_level_information):
            output["em_samples"] = []
            output["f1_samples"] = []
                
        for index in range(batch_size):
            if metadata is not None:
                passage_str = metadata[index]['original_passage']
                offsets = metadata[index]['token_offsets']
                predicted_span = tuple(best_span[index].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output["best_span_str"].append(best_span_string)

                answer_texts = metadata[index].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
                    if (get_sample_level_information):
                        em_sample, f1_sample = bidut.get_em_f1_metrics(best_span_string,answer_texts)
                        output["em_samples"].append(em_sample)
                        output["f1_samples"].append(f1_sample)
                        
        if (get_sample_level_information):
            # Add information about the individual samples for future analysis
            output["span_start_sample_loss"] = []
            output["span_end_sample_loss"] = []
            for i in range (batch_size):
                
                span_start_probs = sum(subresult['span_start_probs'] for subresult in subresults) / len(subresults)
                span_end_probs = sum(subresult['span_end_probs'] for subresult in subresults) / len(subresults)
                span_start_loss = nll_loss(span_start_probs[[i],:], span_start.squeeze(-1)[[i]])
                span_end_loss = nll_loss(span_end_probs[[i],:], span_end.squeeze(-1)[[i]])
                
                output["span_start_sample_loss"].append(float(span_start_loss.detach().cpu().numpy()))
                output["span_end_sample_loss"].append(float(span_end_loss.detach().cpu().numpy()))
        return output
    
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                get_sample_level_information = False,
                get_attentions = False) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.
        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        
        """
        #################### Sample Bayesian weights ##################
        """
        self.sample_posterior()
        
        """
        ################## MASK COMPUTING ########################
        """
                
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None
        
        """
        ###################### EMBEDDING + HIGHWAY LAYER ########################
        """
#        self.cf_a.use_ELMO
        
        if(self.cf_a.Add_Linear_projection_ELMO):
            embedded_question = self._highway_layer(self._linear_projection_ELMO (self._text_field_embedder(question['character_ids'])["elmo_representations"][-1]))
            embedded_passage = self._highway_layer(self._linear_projection_ELMO(self._text_field_embedder(passage['character_ids'])["elmo_representations"][-1]))
        else:
            embedded_question = self._highway_layer(self._text_field_embedder(question['character_ids'])["elmo_representations"][-1])
            embedded_passage = self._highway_layer(self._text_field_embedder(passage['character_ids'])["elmo_representations"][-1])
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        
        """
        ###################### phrase_layer LAYER ########################
        """

        encoded_question = self._dropout_phrase_layer(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout_phrase_layer(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        """
        ###################### Attention LAYER ########################
        """
        
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
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

        modeled_passage = self._dropout_modeling_passage(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)
        
        """
        ###################### Spans LAYER ########################
        """
        
        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout_spans_output(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                   passage_length,
                                                                                   modeling_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        span_end_representation = torch.cat([final_merged_passage,
                                             modeled_passage,
                                             tiled_start_representation,
                                             modeled_passage * tiled_start_representation],
                                            dim=-1)
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._dropout_span_end_encode(self._span_end_encoder(span_end_representation,
                                                                passage_lstm_mask))
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = self._dropout_spans_output(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        
        best_span = bidut.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
                "best_span": best_span,
                }

        # Compute the loss for training.
        if span_start is not None:
            
            span_start_loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            span_end_loss = nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
            loss = span_start_loss + span_end_loss

            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            
            output_dict["loss"] = loss
            output_dict["span_start_loss"] = span_start_loss
            output_dict["span_end_loss"] = span_end_loss
            
        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            if (get_sample_level_information):
                output_dict["em_samples"] = []
                output_dict["f1_samples"] = []
                
            output_dict['best_span_str'] = []
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
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
                    if (get_sample_level_information):
                        em_sample, f1_sample = bidut.get_em_f1_metrics(best_span_string,answer_texts)
                        output_dict["em_samples"].append(em_sample)
                        output_dict["f1_samples"].append(f1_sample)
                        
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            
        if (get_sample_level_information):
            # Add information about the individual samples for future analysis
            output_dict["span_start_sample_loss"] = []
            output_dict["span_end_sample_loss"] = []
            for i in range (batch_size):
                span_start_loss = nll_loss(util.masked_log_softmax(span_start_logits[[i],:], passage_mask[[i],:]), span_start.squeeze(-1)[[i]])
                span_end_loss = nll_loss(util.masked_log_softmax(span_end_logits[[i],:], passage_mask[[i],:]), span_end.squeeze(-1)[[i]])
                
                output_dict["span_start_sample_loss"].append(float(span_start_loss.detach().cpu().numpy()))
                output_dict["span_end_sample_loss"].append(float(span_end_loss.detach().cpu().numpy()))
        if(get_attentions):
            output_dict["C2Q_attention"] = passage_question_attention
            output_dict["Q2C_attention"] = question_passage_attention
            output_dict["simmilarity"] = passage_question_similarity
            
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }
    
    def train_batch(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        
        """
        It is enough to just compute the total loss because the normal weights 
        do not depend on the KL Divergence
        """
        # Now we can just compute both losses which will build the dynamic graph
        
        output = self.forward(question,passage,span_start,span_end,metadata )
        data_loss = output["loss"]
        
        KL_div = self.get_KL_divergence()
        total_loss =  self.combine_losses(data_loss, KL_div)
        
        self.zero_grad()     # zeroes the gradient buffers of all parameters
        total_loss.backward()
        
        if (type(self._optimizer) == type(None)):
            parameters = filter(lambda p: p.requires_grad, self.parameters())
            with torch.no_grad():
                for f in parameters:
                    f.data.sub_(f.grad.data * self.lr )
        else:
#            print ("Training")
            self._optimizer.step()
            self._optimizer.zero_grad()
            
        return output
    
    def fill_batch_training_information(self, training_logger,
                                        output_batch):
        """
        Function to fill the the training_logger for each batch. 
        training_logger: Dictionary that will hold all the training info
        output_batch: Output from training the batch
        """
        training_logger["train"]["span_start_loss_batch"].append(output_batch["span_start_loss"].detach().cpu().numpy())
        training_logger["train"]["span_end_loss_batch"].append(output_batch["span_end_loss"].detach().cpu().numpy())
        training_logger["train"]["loss_batch"].append(output_batch["loss"].detach().cpu().numpy())
        # Training metrics:
        metrics = self.get_metrics()
        training_logger["train"]["start_acc_batch"].append(metrics["start_acc"])
        training_logger["train"]["end_acc_batch"].append(metrics["end_acc"])
        training_logger["train"]["span_acc_batch"].append(metrics["span_acc"])
        training_logger["train"]["em_batch"].append(metrics["em"])
        training_logger["train"]["f1_batch"].append(metrics["f1"])
    
        
    def fill_epoch_training_information(self, training_logger,device,
                                        validation_iterable, num_batches_validation):
        """
        Fill the information per each epoch
        """
        Ntrials_CUDA = 100
        # Training Epoch final metrics
        metrics = self.get_metrics(reset = True)
        training_logger["train"]["start_acc"].append(metrics["start_acc"])
        training_logger["train"]["end_acc"].append(metrics["end_acc"])
        training_logger["train"]["span_acc"].append(metrics["span_acc"])
        training_logger["train"]["em"].append(metrics["em"])
        training_logger["train"]["f1"].append(metrics["f1"])
        
        self.set_posterior_mean(True)
        self.eval()
        
        data_loss_validation = 0
        loss_validation = 0
        with torch.no_grad():
            # Compute the validation accuracy by using all the Validation dataset but in batches.
            for j in range(num_batches_validation):
                tensor_dict = next(validation_iterable)
                
                trial_index = 0
                while (1):
                    try:
                        tensor_dict = pytut.move_to_device(tensor_dict, device) ## Move the tensor to cuda
                        output_batch = self.forward(**tensor_dict)
                        break;
                    except RuntimeError as er:
                        print (er.args)
                        torch.cuda.empty_cache()
                        time.sleep(5)
                        torch.cuda.empty_cache()
                        trial_index += 1
                        if (trial_index == Ntrials_CUDA):
                            print ("Too many failed trials to allocate in memory")
                            send_error_email(str(er.args))
                            sys.exit(0)
                
                data_loss_validation += output_batch["loss"].detach().cpu().numpy() 
                        
                ## Memmory management !!
            if (self.cf_a.force_free_batch_memory):
                del tensor_dict["question"]; del tensor_dict["passage"]
                del tensor_dict
                del output_batch
                torch.cuda.empty_cache()
            if (self.cf_a.force_call_garbage_collector):
                gc.collect()
                
            data_loss_validation = data_loss_validation/num_batches_validation
#            loss_validation = loss_validation/num_batches_validation
    
            # Training Epoch final metrics
        metrics = self.get_metrics(reset = True)
        training_logger["validation"]["start_acc"].append(metrics["start_acc"])
        training_logger["validation"]["end_acc"].append(metrics["end_acc"])
        training_logger["validation"]["span_acc"].append(metrics["span_acc"])
        training_logger["validation"]["em"].append(metrics["em"])
        training_logger["validation"]["f1"].append(metrics["f1"])
        
        training_logger["validation"]["data_loss"].append(data_loss_validation)
        self.train()
        self.set_posterior_mean(False)
    
    def trim_model(self, mu_sigma_ratio = 2):
        
        total_size_w = []
        total_removed_w = []
        total_size_b = []
        total_removed_b = []
        
        if (self.cf_a.VB_Linear_projection_ELMO):
                VBmodel = self._linear_projection_ELMO
                size_w, removed_w, size_b, removed_b = Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                total_size_w.append(size_w)
                total_removed_w.append(removed_w)
                total_size_b.append(size_b)
                total_removed_b.append(removed_b)
                
        if (self.cf_a.VB_highway_layers):
                VBmodel = self._highway_layer._module.VBmodels[0]
                Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                size_w, removed_w, size_b, removed_b = Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                total_size_w.append(size_w)
                total_removed_w.append(removed_w)
                total_size_b.append(size_b)
                total_removed_b.append(removed_b)
                
        if (self.cf_a.VB_similarity_function):
                VBmodel = self._matrix_attention._similarity_function
                Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                size_w, removed_w, size_b, removed_b = Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                total_size_w.append(size_w)
                total_removed_w.append(removed_w)
                total_size_b.append(size_b)
                total_removed_b.append(removed_b)
                
        if (self.cf_a.VB_span_start_predictor_linear):
                VBmodel = self._span_start_predictor_linear
                Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                size_w, removed_w, size_b, removed_b = Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                total_size_w.append(size_w)
                total_removed_w.append(removed_w)
                total_size_b.append(size_b)
                total_removed_b.append(removed_b)
                
        if (self.cf_a.VB_span_end_predictor_linear):
                VBmodel = self._span_end_predictor_linear
                Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                size_w, removed_w, size_b, removed_b = Vil.trim_LinearVB_weights(VBmodel,  mu_sigma_ratio)
                total_size_w.append(size_w)
                total_removed_w.append(removed_w)
                total_size_b.append(size_b)
                total_removed_b.append(removed_b)
                
        
        return  total_size_w, total_removed_w, total_size_b, total_removed_b
#    print (weights_to_remove_W.shape)

    
    """
    BAYESIAN NECESSARY FUNCTIONS
    """
    sample_posterior = GeneralVBModel.sample_posterior
    get_KL_divergence = GeneralVBModel.get_KL_divergence
    set_posterior_mean = GeneralVBModel.set_posterior_mean
    combine_losses = GeneralVBModel.combine_losses
    
    def save_VB_weights(self):
        """
        Function that saves only the VB weights of the model.
        """
        pretrained_dict = ...
        model_dict = self.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
