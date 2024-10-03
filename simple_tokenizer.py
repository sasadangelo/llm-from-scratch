import re
from vocabulary import Vocabulary
from tokenizer import Tokenizer

class SimpleTokenizer(Tokenizer):
  def __init__(self, vocabulary):
    super().__init__(vocabulary)

  def train(self, book_file_name):
    vocabulary = Vocabulary()
    # Add the book to the vocabulary
    vocabulary.add_book(book_file_name)
    self.vocabulary = vocabulary.vocabulary
    # The inverse vocabulary is like the vocabulary but with ids and token inverted.
    #
    # Vocabulary:
    #
    # added	1297
    # addicted	1298
    # adding	1299
    # addition	1300
    # additional	1301
    #
    # Inverse Vocabulary:
    #
    # 1297 added
    # 1298 addicted
    # 1299 adding
    # 1300 addition
    # 1301 additional
    self.inverse_vocabulary ={i:s for s, i in vocabulary.vocabulary.items()}

  def encode(self, text):
    # Split the input text in words
    words_with_spaces =re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # For each word found in the previous step, remove extra spaces. This step remove whitespaces from the sentence.
    words = [item.strip() for item in words_with_spaces if item.strip()]
    # The goal of this line is simply replace the word not present in the vocabulary with the <|unknown|> token.
    words = [item if item in self.vocabulary else "<|unknown|>" for item in words]
    # Convert the token in ids according to the map in the vocabulary.
    ids = [self.vocabulary[s] for s in words]
    return ids

  def decode(self, ids):
    # For each id simply get the relative token from the inverse vocabulary.
    text = " ".join([self.inverse_vocabulary[i] for i in ids])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text