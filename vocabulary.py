import re

class Vocabulary:
    VOCABULARY_FILE_NAME="vocabulary.txt"

    def __init__(self):
        self.vocabulary = {}

    def add_book(self, book_file_name):
        # Open and read the book in txt format
        with open(book_file_name, "r", encoding="utf-8") as f:
            raw_text=f.read()
        # Split the whole book in words, included whitespaces. Each word is a token.
        words_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        # Remove whitespaces as token
        words = [item for item in words_with_spaces if item.strip()]
        # Order and remove duplicated from the token list
        all_tokens = sorted(set(words))
        # Add two extra tokens to the list:
        # - unknown words
        # - endoftext
        all_tokens.extend(["<|endoftext|>", "<|unknown|>"])
        self.vocabulary = {token: integer for integer,token in enumerate(all_tokens)}
        # save the vocabulary in the vocabulary.txt file
        self._save_vocabulary_to_file(Vocabulary.VOCABULARY_FILE_NAME)

    def len(self):
        return len(self.vocabulary)

    def _save_vocabulary_to_file(self, file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            for token, token_id in self.vocabulary.items():
                f.write(f"{token}\t{token_id}\n")