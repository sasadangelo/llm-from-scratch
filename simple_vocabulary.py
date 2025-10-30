import re
from vocabulary import Vocabulary


class SimpleVocabulary(Vocabulary):
    def add_document(self, book_file_name):
        """Adds tokens from a text file to the vocabulary with simple tokenization."""
        # Open and read the book in txt format
        with open(book_file_name, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Split the whole document in words, included whitespaces. Each word is a token.
        words_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

        # Remove whitespaces as token
        words = [item for item in words_with_spaces if item.strip()]

        # Order and remove duplicated from the token list
        all_tokens = sorted(set(words))

        # Add two extra tokens to the list:
        # - unknown words
        # - endoftext
        all_tokens.extend(["<|endoftext|>", "<|unknown|>"])

        # Create the vocabulary as a dictionary {token: token_id}
        self.vocabulary = {token: index for index, token in enumerate(all_tokens)}

        # save the vocabulary in the vocabulary.txt file
        self._save_vocabulary_to_file(Vocabulary.VOCABULARY_FILE_NAME)
