from abc import ABC, abstractmethod


class Vocabulary(ABC):
    VOCABULARY_FILE_NAME = "vocabulary.txt"

    def __init__(self):
        self.vocabulary = {}

    @abstractmethod
    def add_document(self, book_file_name):
        """Abstract method to be implemented by subclasses."""
        pass

    def len(self):
        return len(self.vocabulary)

    def _save_vocabulary_to_file(self, file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            for token, token_id in self.vocabulary.items():
                f.write(f"{token}\t{token_id}\n")
