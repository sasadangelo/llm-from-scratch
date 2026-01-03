# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import os
from tokenizers import Tokenizer, pre_tokenizers, decoders, Regex
from transformers import AutoTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizer.tokenizer import BaseTokenizer

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
    "<|unknown|>",
]

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class HuggingfaceTokenizer(BaseTokenizer):
    def __init__(self):
        self._tokenizer = None

    def train(self, *args, **kwargs):
        """
        Train the tokenizer from data.
        Expects either:
            - kwargs['text']: string
            - kwargs['file']: path to a text file
            - kwargs['vocab_size']: int
        """
        vocab_size = kwargs.get("vocab_size", 3000)
        text = kwargs.get("text", None)
        file_path = kwargs.get("file", None)

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif text is None:
            raise ValueError("train requires either 'text' or 'file' argument")

        tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token="<|unknown|>"))
        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator([text], trainer)
        self._tokenizer = tokenizer

    def encode(self, text, *args, **kwargs):
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        if isinstance(text, str):
            return self._tokenizer.encode(text, add_special_tokens=False).ids
        elif isinstance(text, list):
            return [self._tokenizer.encode(t, add_special_tokens=False).ids for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path):
        if self._tokenizer is None:
            raise ValueError("No tokenizer to save")
        os.makedirs(path, exist_ok=True)
        self._tokenizer.save(os.path.join(path, "tokenizer.json"))

    def load(self, path_or_model_name):
        if os.path.exists(path_or_model_name):
            # Load the tokenizer.json from the local folder in input
            tokenizer_path = os.path.join(path_or_model_name, "tokenizer.json")
            if os.path.isfile(tokenizer_path):
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
            return
        print(f"Loading pretrained tokenizer '{path_or_model_name}' (cached or remote)")
        hf_tok = AutoTokenizer.from_pretrained(path_or_model_name)
        tokenizer_json = hf_tok.backend_tokenizer.to_str()
        self._tokenizer = Tokenizer.from_str(tokenizer_json)

    def get_vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def get_special_tokens(self):
        return SPECIAL_TOKENS
