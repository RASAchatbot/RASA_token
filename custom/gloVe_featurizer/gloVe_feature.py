from __future__ import annotations
import numpy as np
import logging
import typing
from typing import Any, List, Text, Dict, Tuple, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)
from custom.mecab_tokenizer.mecabtok_3 import MecabTokenizer
from rasa.utils.tensorflow.constants import MEAN_POOLING, POOLING
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.training_data import TrainingData


logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER,
    is_trainable=False,
)
class GloveFeaturizer(DenseFeaturizer, GraphComponent):
    """A class that featurizes using Glove."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [MecabTokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            **DenseFeaturizer.get_default_config(),
            # Specify what pooling operation should be used to calculate the vector of
            # the complete utterance. Available options: 'mean' and 'max'
            POOLING: MEAN_POOLING,
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["konlpy"]

    def __init__(
        self, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> None:
        """Instantiates a new `MitieFeaturizer` instance."""
        super().__init__(execution_context.node_name, config)
        self.pooling_operation = self._config[POOLING]
        self.glove_dict = loadGloVe('custom\\gloVe_featurizer\\gloVe_data\\glove.txt')

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GloveFeaturizer:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, execution_context)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass

    # def ndim(self, feature_extractor: "mitie.total_word_feature_extractor") -> int:
    #     """Returns the number of dimensions."""
    #     return feature_extractor.num_dimensions

    def process(self, messages: List[Message]) -> List[Message]:
        """Featurizes all given messages in-place.

        Returns:
          The given list of messages which have been modified in-place.
        """
        for message in messages:
            #self._process_message(message)
            self._set_features(message, 'text')
        return messages

    def process_training_data(
        self, training_data: TrainingData
    ) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: Training data.
          model: A Mitie model.

        Returns:
          Same training data after processing.
        """
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                # attribute : text or response
                #   text : intent examples
                #   response : system utter[action] examples
                # featurizing only text
                if attribute == 'text':
                    self._set_features(example, attribute)

        #self.process(training_data.training_examples)
        return training_data

    def _process_message(self, message: Message) -> None:
        """Processes a message."""

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            self._set_features(message, attribute)
            # self._process_training_example(
            #     message, attribute
            # )

    def _process_training_example(
        self,
        example: Message,
        attribute: Text,
    ) -> None:
        tokens = example.get(TOKENS_NAMES[attribute])
        if self.glove_dict is None:
            logger.warning('model is None')
            raise ValueError('Model이 없다!!')

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            # attribute : text or response
            #   text : intent examples
            #   response : system utter[action] examples
            # featurizing only text
            if attribute == 'text':
                self._set_features(tokens, attribute)


    def _set_features(
        self,
        message: Message,
        attribute: Text
    ) -> None:

        tokens = message.get(TOKENS_NAMES[attribute])

        # start_time = time.time()

        # convert tokens object to tokens text
        # for inputting to featurizer
        tokens_text = self._tokens_to_text(tokens)

        # 실제 dense feature 뽑는 메서드
        features = self.features_for_tokens(tokens_text)

        # features = self._combine_with_existing_dense_features(
        #     message, features, FEATURIZER_CLASS_ALIAS[attribute]
        # )

        #message.set(DENSE_FEATURIZABLE_ATTRIBUTES[attribute], features)

        final_features = Features(
            features,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        # message.add_features(final_sequence_features)
        #
        # final_sentence_features = Features(
        #     sentence_features,
        #     FEATURE_TYPE_SENTENCE,
        #     attribute,
        #     self._config[FEATURIZER_CLASS_ALIAS],
        # )
        message.add_features(final_features)

    def features_for_tokens(
        self,
        tokens: List[Token],
    ) -> np.asarray:
        """Calculates features."""
        '''
               딕셔너리 형태의 glove_dict으로부터 각 토큰의 임베딩 값을 쌓는다.
               '''
        res = []
        for tok in tokens:
            try:
                res.append(self.glove_dict[tok])
            except KeyError:
                res.append(self.glove_dict['<unk>'])
        embs = np.asarray(res)

        logger.debug('embs shape : {}'.format(embs.shape))
        logger.debug('embs : {}'.format(embs))

        return embs

        # sequence_features = np.array(
        #     [feature_extractor.get_feature_vector(token.text) for token in tokens]
        # )
        #
        # sentence_fetaures = self.aggregate_sequence_features(
        #     sequence_features, self.pooling_operation
        # )
        #
        # return sequence_features, sentence_fetaures

    @staticmethod
    def _tokens_to_text(tokens: List[Token]) -> List[Text]:
        text = []
        for token in tokens:
            text.append(token.text)
        return text

def loadGloVe(glove_path):
    embedding_dict = dict()
    f = open(glove_path, encoding="utf8")

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')  # 100개의 값을 가지는 array로 변환
        embedding_dict[word] = word_vector_arr
    f.close()
    logger.debug('%s개의 Embedding vector가 있습니다.' % len(embedding_dict))
    return embedding_dict