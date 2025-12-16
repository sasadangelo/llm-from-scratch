# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import fitz
import re


class BookSource:
    def __init__(self, path: str, skip_pages: str | None = None):
        self.path = path
        self.skip_pages = self._parse_page_spec(skip_pages)

    def load(self) -> str:
        pages = self._read_pages()

        filtered_pages = [text for idx, text in enumerate(pages) if idx not in self.skip_pages]

        text = "\n".join(filtered_pages)
        return self._clean_text(text)

    def _read_pages(self) -> list[str]:
        doc = fitz.open(self.path)
        return [page.get_text() for page in doc]

    # def _clean_text(self, text: str) -> str:
    #     text = text.replace("\r", "\n")
    #     text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    #     text = re.sub(r"[ \t]+", " ", text)
    #     text = re.sub(r"\n{3,}", "\n\n", text)
    #     return text.strip()

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)  # remove page numbers
        text = re.sub(r"[ \t]+", " ", text)  # normalize spaces
        # strip trailing spaces at the end of each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multiple newlines
        return text.strip()

    def _parse_page_spec(self, spec: str | None) -> set[int]:
        """
        Converts a page spec like '1,2-10,15' into a set of 0-based indices.
        """
        if not spec:
            return set()

        pages = set()

        for part in spec.split(","):
            part = part.strip()

            if "-" in part:
                start, end = part.split("-")
                start, end = int(start), int(end)
                for p in range(start, end + 1):
                    pages.add(p - 1)
            else:
                pages.add(int(part) - 1)

        return pages
