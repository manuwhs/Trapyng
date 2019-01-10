import json
import logging
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.iterators import BucketIterator
import numpy as np
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# @DatasetReader.register("squad1")
class Squad1Reader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tokenizer_indexer_type = "") -> None:
        super().__init__(lazy)
        
        if (tokenizer_indexer_type == "elmo"):
            tokenizer = WordTokenizer()
            token_indexers = {'character_ids': ELMoTokenCharactersIndexer()} 
        else:
            tokenizer = tokenizer or WordTokenizer()
            token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
            
        self._tokenizer = tokenizer 
        self._token_indexers = token_indexers


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     zip(span_starts, span_ends),
                                                     answer_texts,
                                                     tokenized_paragraph)
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return util.make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts)
        
def load_SQUAD1_dataset(cf_a,vocab):
    """
    Loads the dataset and creates iterators and so on
    """
    ## Create the Data Reader with the Tokenization and indexing
    if (cf_a.datareader_lazy):
        #If we do lazy loading, the training will be slower but we dont have RAM so....
        # We also can specify:
        instances_per_epoch_train = cf_a.instances_per_epoch_train
        instances_per_epoch_validation = cf_a.instances_per_epoch_validation
        max_instances_in_memory = cf_a.max_instances_in_memory 
    else:
        instances_per_epoch_train = None
        instances_per_epoch_validation = None
        max_instances_in_memory = None
    
    ## Instantiate the datareader
    squad_reader = Squad1Reader(lazy = cf_a.datareader_lazy, 
                                tokenizer_indexer_type = cf_a.tokenizer_indexer_type)
    
    ## Load the datasets
    train_dataset = squad_reader.read(file_path = cf_a.train_squad1_file)
    validation_dataset =  squad_reader.read(file_path = cf_a.validation_squad1_file)
    """
    ########################## ITERATORS  ############################
    Iterator that will get the samples for the problem
    """

    if(cf_a.datareader_lazy == False):
        instances_per_epoch_train = len(train_dataset)
        instances_per_epoch_validation = len(validation_dataset)
    
    train_iterator = BucketIterator(batch_size= cf_a.batch_size_train, instances_per_epoch = instances_per_epoch_train,
                              max_instances_in_memory = max_instances_in_memory,
                              sorting_keys=[["passage", "num_tokens"], ["question", "num_tokens"]])
    train_iterator.index_with(vocab)
    
    validation_iterator = BucketIterator(batch_size= cf_a.batch_size_validation, instances_per_epoch = instances_per_epoch_validation,
                              max_instances_in_memory = max_instances_in_memory,
                              sorting_keys=[["passage", "num_tokens"], ["question", "num_tokens"]])
    
    validation_iterator.index_with(vocab)
    
    num_batches = int(np.ceil(instances_per_epoch_train/cf_a.batch_size_train))
    num_batches_validation = int(np.ceil(instances_per_epoch_validation/cf_a.batch_size_validation))
    
    # Create the iterator over the data:
    train_iterable = train_iterator(train_dataset)
    validation_iterable = validation_iterator(validation_dataset)
    
    return squad_reader, num_batches, train_iterable, num_batches_validation, validation_iterable