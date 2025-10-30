# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the tokenizer from data"""
        pass

    @abstractmethod
    def encode(self, text, *args, **kwargs):
        """Encode text to list of token ids"""
        pass

    @abstractmethod
    def decode(self, ids):
        """Decode list of token ids to text"""
        pass

    @abstractmethod
    def save(self, path):
        """Save tokenizer to disk"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load tokenizer from disk"""
        pass

    @abstractmethod
    def get_vocab_size(self):
        """Return size of the vocabulary"""
        pass

    @abstractmethod
    def get_special_tokens(self):
        """Return list of special tokens"""
        pass
