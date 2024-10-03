from abc import ABC, abstractmethod

class Tokenizer(ABC):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary.vocabulary
        self.inverse_vocabulary = {i: s for s, i in vocabulary.vocabulary.items()}

    @abstractmethod
    def train(self, book_file_name):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, ids):
        pass