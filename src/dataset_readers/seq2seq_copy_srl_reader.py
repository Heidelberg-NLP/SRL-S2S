import logging
from typing import List, Dict
import json

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, SequenceLabelField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("my_copynet_seq2seq")
class SRLCopyNetDatasetReader(DatasetReader):
    """
    Reads a JSON file containing all necessary information, and create a dataset suitable for a
    ``SRLCopyNetSeq2Seq`` model and is based on the CopyNetDatasetReader code.
    Each input line in the file should cointain at least the following:

        { "seq_words":      ["List", "of", "input", "sentence" "having" "one", "predicate", "of", "interest"],
          "BIO":            ["O", "O", "O", "O" "B-V" "O", "O", "O", "O"],
          "pred_sense":     [4, "having", "optional.verb.sense", "V"],
          "seq_tag_tokens": ["This", "list", "(#", "contains", "V)" ,"labeled", "(#", "words", "A0)", "during", "training"],
          "src_lang":       "<DE>",
          "tgt_lang":       "<DE-SRL>"
        }

    An instance produced by ``SRLCopyNetDatasetReader`` will containing at least the following fields:
    - ``source_tokens``: a ``TextField`` containing the tokenized source sentence,
       including the ``START_SYMBOL`` and ``END_SYMBOL``.
       This will result in a tensor of shape ``(batch_size, source_length)``.
    - ``source_token_ids``: an ``ArrayField`` of size ``(batch_size, trimmed_source_length)``
      that contains an ID for each token in the source sentence. Tokens that
      match at the lowercase level will share the same ID. If ``target_tokens``
      is passed as well, these IDs will also correspond to the ``target_token_ids``
      field, i.e. any tokens that match at the lowercase level in both
      the source and target sentences will share the same ID. Note that these IDs
      have no correlation with the token indices from the corresponding
      vocabulary namespaces.
    - ``verb_indicator``: a ``SequenceLabelField`` indicating in which index is the predicate that will be tagged.
    - ``source_to_target``: a ``NamespaceSwappingField`` that keeps track of the index
      of the target token that matches each token in the source sentence.
      When there is no matching target token, the OOV index is used.
      This will result in a tensor of shape ``(batch_size, trimmed_source_length)``.
    - ``metadata``: a ``MetadataField`` which contains the source tokens and
      potentially target tokens as lists of strings and several useful information for evaluation of outputs.
    When ``target_string`` is passed, the instance will also contain these fields:
    - ``target_tokens``: a ``TextField`` containing the tokenized target sentence,
      including the ``START_SYMBOL`` and ``END_SYMBOL``. This will result in
      a tensor of shape ``(batch_size, target_length)``.
    - ``target_token_ids``: an ``ArrayField`` of size ``(batch_size, target_length)``.
      This is calculated in the same way as ``source_token_ids``.
    See the "Notes" section below for a description of how these fields are used.

    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    available_languages : ``Dict[str, int]``, optional
        A dictionary mapping language keys to indices. This will be passed to the Model to
         construct the Language Indicator Embeddings.
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.

    Notes
    -----
    By ``source_length`` we are referring to the number of tokens in the source
    sentence including the ``START_SYMBOL`` and ``END_SYMBOL``, while
    ``trimmed_source_length`` refers to the number of tokens in the source sentence
    *excluding* the ``START_SYMBOL`` and ``END_SYMBOL``, i.e.
    ``trimmed_source_length = source_length - 2``.
    On the other hand, ``target_length`` is the number of tokens in the target sentence
    *including* the ``START_SYMBOL`` and ``END_SYMBOL``.
    In the context where there is a ``batch_size`` dimension, the above refer
    to the maximum of their individual values across the batch.
    In regards to the fields in an ``Instance`` produced by this dataset reader,
    ``source_token_ids`` and ``target_token_ids`` are primarily used during training
    to determine whether a target token is copied from a source token (or multiple matching
    source tokens), while ``source_to_target`` is primarily used during prediction
    to combine the copy scores of source tokens with the generation scores for matching
    tokens in the target namespace.
    """

    def __init__(self,
                 target_namespace: str,
                 available_languages: Dict[str, int] = None,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._available_languages = available_languages or {"<EN>": 0,
                                                             "<EN-SRL>": 1,
                                                             "<DE>": 2,
                                                             "<DE-SRL>": 3,
                                                             "<FR>": 4,
                                                             "<FR-SRL>": 5}
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._source_token_indexers or \
                not isinstance(self._source_token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CopyNetDatasetReader expects 'source_token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        self._target_token_indexers: Dict[str, TokenIndexer] = {
                "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = json.loads(line)
                if not line: continue
                yield self.text_to_instance("seq_words", "seq_tag_tokens", line)

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    @overrides
    def text_to_instance(self, source_key: str, target_key: str = None, line_obj: Dict = {}) -> Instance:
        """
        Turn json object into an ``Instance``.
        Parameters
        ----------
        source_key : ``str``, required, json object key name of the source sequence
        target_key : ``str``, optional (default = None), json object key name of the target sequence
        line_obj : ``Dict``, required, json object containing the raw instance info
        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """

        # Read source and target
        target_sequence = line_obj.get(target_key, None)
        lang_src_token = line_obj["src_lang"].upper()
        lang_tgt_token = line_obj["tgt_lang"].upper()

        # Read Predicate Indicator and make Array
        verb_label = [0, 0] + [1 if label[-2:] == "-V" else 0 for label in line_obj["BIO"]] + [0]

        # Read Language Indicator and make Array
        lang_src_ix = self._available_languages[lang_src_token]
        lang_tgt_ix = self._available_languages[lang_tgt_token]
        # This array goes to the encoder as a whole
        lang_src_ix_arr = [0, 0] + [lang_src_ix for tok in line_obj[source_key]] + [0]
        # This array goes to each one of the decoder_steps
        lang_tgt_ix_arr = lang_tgt_ix # is just int for step decoder dimensionality

        # Tokenize Source
        tokenized_source = list(map(Token, line_obj[source_key])) # Data comes already tokenized!
        tokenized_source.insert(0, Token(lang_tgt_token))
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        # Process Target info during training...
        if target_sequence is not None:
            tokenized_target = list(map(Token, line_obj[target_key]))
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        # Add Verb Indicator to the Fields
        fields_dict['verb_indicator'] = SequenceLabelField(verb_label, source_field)
        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokenized_source[verb_label.index(1)].text
        meta_fields["verb"] = verb

        # Add Language Indicator to the Fields
        meta_fields["src_lang"] = lang_src_token
        meta_fields["tgt_lang"] = lang_tgt_token
        meta_fields["original_BIO"] = line_obj.get("BIO", [])
        meta_fields["original_predicate_senses"] = line_obj.get("pred_sense_origin", [])
        meta_fields["predicate_senses"] = line_obj.get("pred_sense", [])
        meta_fields["original_target"] = line_obj.get("seq_tag_tokens", [])
        fields_dict['language_enc_indicator'] = ArrayField(np.array(lang_src_ix_arr))
        fields_dict['language_dec_indicator'] = ArrayField(np.array(lang_tgt_ix_arr))

        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)
