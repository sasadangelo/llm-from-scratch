# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
# PYTHONPATH=. python3 -m pytest -v llmtest/test_datasets/test_booksource.py
# -----------------------------------------------------------------------------
# --- Suppress deprecation warnings from PyMuPDF ---
# flake8: noqa: E402
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import fitz
import os
from datasets.sources.book_source import BookSource


class TestBookSource(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a small test PDF with 5 pages
        cls.test_pdf = "test_book.pdf"
        doc = fitz.open()
        for i in range(5):
            page = doc.new_page()
            page.insert_text((72, 72), f"This is page {i+1}\nSome sample text.")
        doc.save(cls.test_pdf)
        doc.close()

    @classmethod
    def tearDownClass(cls):
        # Remove test PDF after tests
        if os.path.exists(cls.test_pdf):
            os.remove(cls.test_pdf)

    def test_load_without_skip(self):
        book = BookSource(path=self.test_pdf)
        text = book.load()
        # Check that all pages are present
        self.assertIn("This is page 1", text)
        self.assertIn("This is page 5", text)

    def test_load_with_skip(self):
        # Skip pages 2 and 4 (0-based index inside class)
        book = BookSource(path=self.test_pdf, skip_pages="2,4")
        text = book.load()
        # Skipped pages should not appear
        self.assertNotIn("This is page 2", text)
        self.assertNotIn("This is page 4", text)
        # Other pages should still appear
        self.assertIn("This is page 1", text)
        self.assertIn("This is page 3", text)
        self.assertIn("This is page 5", text)

    def test_clean_text_removes_extra_whitespace(self):
        book = BookSource(path=self.test_pdf)
        raw_text = "Line1\rLine2  \n\n\nLine3"
        cleaned = book._clean_text(raw_text)
        self.assertEqual(cleaned, "Line1\nLine2\n\nLine3")


if __name__ == "__main__":
    unittest.main()
