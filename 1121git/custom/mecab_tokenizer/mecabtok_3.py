from __future__ import annotations
from konlpy.tag  import Mecab

from typing import List, Text, Dict, Any
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

from rasa.shared.utils.io import DEFAULT_ENCODING


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class MecabTokenizer(Tokenizer):
    """Tokenizes messages using the `mecab` library.."""

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the Tokenizer framework."""
        super().__init__(component_config)
        self.tokenizer = Mecab('C:\\mecab\\mecab-ko-dic')
    
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default config (see parent class for full docstring)."""
        return {
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["konlpy"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MecabTokenizer:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        from konlpy.tag import Mecab

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

    def _token_from_offset(
        self, text: bytes, offset: int, encoded_sentence: bytes
    ) -> Token:
        return Token(
            text.decode(DEFAULT_ENCODING),
            self._byte_to_char_offset(encoded_sentence, offset),
        )

    @staticmethod
    def _byte_to_char_offset(text: bytes, byte_offset: int) -> int:
        return len(text[:byte_offset].decode(DEFAULT_ENCODING))
