# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import os
import pickle
from functools import lru_cache
from tokenizer.tokenizer import BaseTokenizer
import rustbpe
import tiktoken

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


class RustBPETokenizer(BaseTokenizer):
    """Tokenizer is trained with rustbpe and used via tiktoken for encoding/decoding"""

    # def __init__(self, enc, bos_token="<|bos|>"):
    #     self.enc = enc
    #     self.bos_token_id = self.encode_special(bos_token)

    # -----------------------
    # Train / Load
    # -----------------------
    def train(self, *args, **kwargs):
        vocab_size = kwargs.get("vocab_size", 3000)
        text_iterator = kwargs.get("text_iterator")
        file_path = kwargs.get("file")

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text_iterator = [f.read()]
        elif text_iterator is None:
            raise ValueError("train requires 'file' or 'text_iterator'")

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

        enc = tiktoken.Encoding(
            name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens
        )
        self.enc = enc
        self.bos_token_id = self.encode_special("<|bos|>")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        pickle_path = os.path.join(path, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)

    @classmethod
    def load(cls, path):
        pickle_path = os.path.join(path, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    # -----------------------
    # Encode / Decode
    # -----------------------
    def encode(self, text, *args, **kwargs):
        prepend = kwargs.get("prepend")
        append = kwargs.get("append")
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            if append is not None:
                for row in ids:
                    row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    @lru_cache(maxsize=32)
    def encode_special(self, token):
        return self.enc.encode_single_token(token)

    def get_bos_token_id(self):
        return self.bos_token_id
