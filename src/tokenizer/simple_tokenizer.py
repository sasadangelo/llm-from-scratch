# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import re
import json
from tokenizer.tokenizer import BaseTokenizer

UNKNOWN_TOKEN = "<|unknown|>"


class SimpleTokenizer(BaseTokenizer):
    def __init__(self, vocabulary=None):
        self._vocabulary = vocabulary if vocabulary else {}
        self._inverse_vocabulary = {i: t for t, i in self._vocabulary.items()}

    def train(self, *args, **kwargs):
        """
        Train tokenizer from data.
        Expects either:
            - kwargs['text']: a string
            - kwargs['file']: path to a text file
        """
        text = kwargs.get("text", None)
        file_path = kwargs.get("file", None)

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif text is None:
            raise ValueError("train requires either 'text' or 'file' argument")

        # split text into words
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [w.strip() for w in words if w.strip()]

        # build vocabulary: assign incremental ids
        for idx, word in enumerate(sorted(set(words)), start=0):
            self._vocabulary[word] = idx

        # add unknown token
        if UNKNOWN_TOKEN not in self._vocabulary:
            self._vocabulary[UNKNOWN_TOKEN] = len(self._vocabulary)

        # inverse map
        self._inverse_vocabulary = {i: t for t, i in self._vocabulary.items()}

    def encode(self, text, *args, **kwargs):
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [w.strip() for w in words if w.strip()]
        words = [w if w in self._vocabulary else UNKNOWN_TOKEN for w in words]
        return [self._vocabulary[w] for w in words]

    def decode(self, ids):
        words = [self._inverse_vocabulary.get(i, UNKNOWN_TOKEN) for i in ids]
        text = " ".join(words)
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocabulary, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self._vocabulary = json.load(f)
            self._inverse_vocabulary = {i: t for t, i in self._vocabulary.items()}

    def get_vocab_size(self):
        return len(self._vocabulary)

    def get_special_tokens(self):
        return [UNKNOWN_TOKEN]
