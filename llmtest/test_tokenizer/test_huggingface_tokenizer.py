# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
# PYTHONPATH=. python3 -m pytest -v llmtest/test_tokenizer/test_huggingface_tokenizer.py
# -----------------------------------------------------------------------------
import os
import tempfile
import unittest
from tokenizer import HuggingfaceTokenizer


class TestHuggingfaceTokenizer(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._save_dir = os.path.join(self._tmp_dir.name, "llm")
        self._train_file = "robinson-crusoe.txt"
        self._tokenizer = HuggingfaceTokenizer()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_encode(self):
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        input_text = "Robinson Crusoe is a book by Daniel Defoe"

        token_ids = self._tokenizer.encode(input_text)
        print("Token IDs:", token_ids)

        # Expected IDs attesi (output noto)
        expected_ids = [4857, 2259, 407, 268, 3153, 397, 8829, 85, 8825, 1890]

        # 4Validations
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        self.assertEqual(token_ids, expected_ids, "Token IDs non corrispondono a quelli attesi")

    def test_decode(self):
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        expected_text = "Robinson Crusoe is a book by Daniel Defoe"

        token_ids = [4857, 2259, 407, 268, 3153, 397, 8829, 85, 8825, 1890]
        output_text = self._tokenizer.decode(token_ids)
        print("Token IDs:", token_ids)

        # Expected IDs attesi (output noto)

        # Validations
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        self.assertEqual(expected_text, output_text, "The decoded text doesn't match")

    def test_train_save_load_decode(self):
        # Train
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        # Save
        self._tokenizer.save(self._save_dir)
        self.assertTrue(os.path.exists(self._save_dir), f"the destination folder {self._save_dir} doesn't exist")
        saved_files = os.listdir(self._save_dir)
        self.assertGreater(len(saved_files), 0, "The destination folder is empty")

        # Load
        loaded_tokenizer = HuggingfaceTokenizer()
        loaded_tokenizer.load(self._save_dir)

        # Decode degli ID noti
        token_ids = [4857, 2259, 407, 268, 3153, 397, 8829, 85, 8825, 1890]
        decoded_text = loaded_tokenizer.decode(token_ids)

        # Validations
        expected_text = "Robinson Crusoe is a book by Daniel Defoe"
        self.assertIsInstance(decoded_text, str)
        self.assertEqual(decoded_text, expected_text, "The decoded text doesn't match")
        print("Decoded text:", decoded_text)
        print("Token IDs:", token_ids)

    def test_pretrained_tokenizer_integration(self):
        """
        Test integration with a pretrained Hugging Face tokenizer and model.
        Uses a lightweight model: distilbert-base-uncased.
        """
        self._tokenizer.load("distilbert-base-uncased")

        input_text = "Robinson Crusoe is a book by Daniel Defoe"

        token_ids = self._tokenizer.encode(input_text)
        print("Token IDs:", token_ids)

        # Expected IDs attesi (output noto)
        expected_ids = [6157, 13675, 26658, 2063, 2003, 1037, 2338, 2011, 3817, 13366, 8913]

        # 4Validations
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        self.assertEqual(token_ids, expected_ids, "Token IDs non corrispondono a quelli attesi")
