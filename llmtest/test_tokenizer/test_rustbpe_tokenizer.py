# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
# PYTHONPATH=. python3 -m pytest -v llmtest/test_tokenizer/test_rustbpe_tokenizer.py
# -----------------------------------------------------------------------------
import os
import tempfile
import unittest
from tokenizer import RustBPETokenizer


class TestRustbpeTokenizer(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._save_file = os.path.join(self._tmp_dir.name, "tokenizer.json")
        self._train_file = "robinson-crusoe.txt"
        self._tokenizer = RustBPETokenizer()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_encode(self):
        self._tokenizer.train(file=self._train_file, vocab_size=10000)

        input_text = "Robinson Crusoe is a book by Daniel Defoe"

        token_ids = self._tokenizer.encode(input_text)
        print("Token IDs:", token_ids)

        # Expected IDs attesi (output noto)
        expected_ids = [4848, 2249, 397, 258, 3143, 387, 8819, 108, 8815, 1882]

        # 4Validations
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        self.assertEqual(token_ids, expected_ids, "Token IDs doesn't match with the expected one")
