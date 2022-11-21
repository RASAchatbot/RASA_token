from konlpy.tag  import Mecab

import abc
import logging
import re

from typing import Dict, Text, List, Any, Optional, Type


from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, MESSAGE_ATTRIBUTES
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_RESPONSE_KEY,
    RESPONSE_IDENTIFIER_DELIMITER,
    ACTION_NAME,
)
import rasa.shared.utils.io
from rasa.shared.nlu.constants import (
    RESPONSE,
    TEXT,
    INTENT,
)

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    component_types=[DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER],
    is_trainable=False,
)

class MecabTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the Tokenizer framework."""
        super().__init__(component_config)
        logger.info('component_config : {}'.format(self.component_config))
        self.tokenizer = Mecab()

    @classmethod
    def required_packages(cls) -> List[Text]:
        # mecab 버전이 다양한데 여기서는 konlpy 내부에서 사용하는 mecab을 사용
        return ['konlpy']

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        #             주격    보격   관형격   목적격   부사격   호격   인용격  접속 조사
        exceptList = ['JKS','JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JC']
        text = message.get(attribute)
        tokenized = self.tokenizer.morphs(text)
        # remove '▁'
        tokenized_pp = []
        for t in tokenized:
            if not t == '▁':
                tokenized_pp.append(t.replace('▁', ''))

        # The token class consists of the tokenized word and the word offset.
        return self._convert_words_to_tokens(tokenized_pp, text)

    def train(
        self,
        training_data: TrainingData
        ) -> Resource:
        """Tokenize all training data."""

        for example in training_data.training_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                if example.get(attribute) is not None:
                    if attribute == INTENT:
                        tokens = self._split_intent(example)
                    else:
                        tokens = self.tokenize(example, attribute)
                        tokens = self.add_cls_token(tokens, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)
    
    def process(self, messages: List[Message]) -> List[Message]:
        """Tokenize the incoming message."""

        tokens = self.tokenize(messages, TEXT)
        tokens = self.add_cls_token(tokens, TEXT)
        messages.set(TOKENS_NAMES[TEXT], tokens)

    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        running_offset = 0
        tokens = []

        for word in words:
            # word_offset = text.index(word, running_offset) # error 원인
            word_offset = text.find(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        logger.debug('tokens : {}'.format(tokens))
        return tokens