# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
# PYTHONPATH=. python3 -m pytest -v llmtest/test_tokenizer/test_simple_tokenizer.py
# -----------------------------------------------------------------------------
import os
import tempfile
import unittest
from tokenizer import SimpleTokenizer


class TestSimpleTokenizer(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._save_file = os.path.join(self._tmp_dir.name, "tokenizer.json")
        self._train_file = "robinson-crusoe.txt"
        self._tokenizer = SimpleTokenizer()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_encode(self):
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        input_text = "Robinson Crusoe is a book by Daniel Defoe"

        token_ids = self._tokenizer.encode(input_text)
        print("Token IDs:", token_ids)

        # Expected IDs attesi (output noto)
        expected_ids = [910, 332, 4823, 1199, 1918, 2058, 349, 356]

        # 4Validations
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        self.assertEqual(token_ids, expected_ids, "Token IDs non corrispondono a quelli attesi")

    def test_decode(self):
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        expected_text = "Robinson Crusoe is a book by Daniel Defoe"

        token_ids = [910, 332, 4823, 1199, 1918, 2058, 349, 356]
        output_text = self._tokenizer.decode(token_ids)
        print("Token IDs:", token_ids)

        # Expected IDs attesi (output noto)

        # 4Validations
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        self.assertEqual(expected_text, output_text, "The decoded text doesn't match")

    def test_train_save_load_decode(self):
        # Train
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        # Save
        self._tokenizer.save(self._save_file)
        self.assertTrue(os.path.exists(self._save_file), f"the destination file {self._save_file} doesn't exist")

        # Load
        loaded_tokenizer = SimpleTokenizer.load(self._save_file)

        # Decode degli ID noti
        token_ids = [910, 332, 4823, 1199, 1918, 2058, 349, 356]
        decoded_text = loaded_tokenizer.decode(token_ids)

        # Validations
        expected_text = "Robinson Crusoe is a book by Daniel Defoe"
        self.assertIsInstance(decoded_text, str)
        self.assertEqual(decoded_text, expected_text, "The decoded text doesn't match")
        print("Decoded text:", decoded_text)
        print("Token IDs:", token_ids)
