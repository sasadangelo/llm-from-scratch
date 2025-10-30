import re
from vocabulary import Vocabulary
from functools import lru_cache


class GPT2Vocabulary(Vocabulary):
    def add_document(self, book_file_name):
        """Adds tokens from a text file to the vocabulary using Byte Pair Encoding (BPE)."""
        # Load the raw text from the book
        with open(book_file_name, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Perform Byte Pair Encoding tokenization (simplified illustration)
        # Tokenize the text into characters and apply BPE merges.
        words = list(raw_text)  # Initially split into characters

        # Example BPE merge logic (real BPE would need to identify and merge frequent pairs)
        # For the sake of this example, we can simulate it by merging common letter pairs.
        bpe_tokens = self.byte_pair_encoding(words)

        # Add the BPE tokens to the vocabulary
        all_tokens = sorted(set(bpe_tokens))
        all_tokens.extend(["<|endoftext|>", "<|unknown|>"])

        # Create the vocabulary with token IDs
        self.vocabulary = {token: index for index, token in enumerate(all_tokens)}

        # Save the vocabulary to a file
        self._save_vocabulary_to_file(Vocabulary.VOCABULARY_FILE_NAME)

    def byte_pair_encoding(self, tokens):
        """A simplified version of the Byte Pair Encoding (BPE) algorithm."""
        # Here we should implement the BPE logic, merging the most frequent symbol pairs.
        # For simplicity, let's simulate a few manual merges (e.g., merging common letters)

        # This is a placeholder logic for actual BPE. In real use, we'd identify pairs to merge.
        merged_tokens = tokens  # No actual merging happens here, it's a simplified example
        return merged_tokens


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """

    # The range from "!" (ASCII code 33) to "~" (ASCII code 126) represents the commonly printable characters in extended ASCII.
    # The range from "¡" (ASCII code 161) to "¬" (ASCII code 172) represents special characters used in languages other than English..
    # The range from "®" (ASCII code 174) to "ÿ" (ASCII code 255) Includes special symbols, punctuation marks, and letters with diacritical
    # marks typical of extended alphabets.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    print("bs: ", bs)
    print("cs: ", cs)
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as a tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


if __name__ == "__main__":
    word = "Robinson"
    print(get_pairs(word))
    print(bytes_to_unicode())
    print(chr(255))
