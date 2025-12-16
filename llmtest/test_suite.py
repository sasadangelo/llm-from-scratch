# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
# PYTHONPATH=. python3 -m pytest -v llmtest/test_suite.py
# -----------------------------------------------------------------------------
import unittest
from llmtest.test_tokenizer import TestHuggingfaceTokenizer, TestRustbpeTokenizer, TestSimpleTokenizer
from llmtest.test_datasets import TestBookSource, TestLanguageModelingDataset


def suite():
    test_suite = unittest.TestSuite()
    # Test Tokenizers
    test_suite.addTest(TestHuggingfaceTokenizer)
    test_suite.addTest(TestRustbpeTokenizer)
    test_suite.addTest(TestSimpleTokenizer)
    # Test Datasets
    test_suite.addTest(TestBookSource)
    test_suite.addTest(TestLanguageModelingDataset)
