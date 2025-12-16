# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
# PYTHONPATH=. python3 -m pytest -v llmtest/test_datasets/test_lm_dataset.py
# -----------------------------------------------------------------------------
import unittest
import torch
from tokenizer import BaseTokenizer
from datasets import LanguageModelingDataset


class DummyTokenizer(BaseTokenizer):
    """Minimal dummy tokenizer for testing purposes."""

    def encode(self, text: str):
        # Trasforma ogni carattere in un intero fittizio
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join([chr(t) for t in tokens])

    def get_special_tokens(self):
        return []

    def get_vocab_size(self):
        return 128  # dummy vocab size

    def load(self, path):
        pass

    def save(self, path):
        pass

    def train(self, texts):
        pass


class TestLanguageModelingDataset(unittest.TestCase):
    def setUp(self):
        self.text = "hello world"
        self.tokenizer = DummyTokenizer()
        self.context_length = 5
        self.stride = 1
        self.dataset = LanguageModelingDataset(
            text=self.text, tokenizer=self.tokenizer, context_length=self.context_length, stride=self.stride
        )

    def test_len(self):
        expected_len = (len(self.tokenizer.encode(self.text)) - self.context_length - 1) // self.stride
        self.assertEqual(len(self.dataset), expected_len)

    def test_getitem_shapes(self):
        input_ids, target_ids = self.dataset[0]
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(target_ids, torch.Tensor)
        self.assertEqual(input_ids.shape[0], self.context_length)
        self.assertEqual(target_ids.shape[0], self.context_length)

    def test_shift_by_one(self):
        input_ids, target_ids = self.dataset[0]
        # target dovrebbe essere input shiftato di 1
        self.assertTrue(torch.equal(target_ids[:-1], input_ids[1:]))


if __name__ == "__main__":
    unittest.main()
