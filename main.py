# main.py
from tokenizer.huggingface_tokenizer import HuggingfaceTokenizer


def main():
    # ---------------------------
    # 1) Train the tokenizer
    # ---------------------------
    # tokenizer = SimpleTokenizer()
    tokenizer = HuggingfaceTokenizer()
    tokenizer.train(file="robinson-crusoe.txt", vocab_size=10000)

    # Salva il vocabolario in un file JSON
    tokenizer.save("llm")
    print(f"Tokenizer trained and saved. Vocab size: {tokenizer.get_vocab_size()}")

    # ---------------------------
    # 2) Encode a fixed string
    # ---------------------------
    input_text = "Robinson Crusoe is a book by Daniel Defoe"

    token_ids = tokenizer.encode(input_text)
    print("Token IDs:", token_ids)

    # ---------------------------
    # 3) Decode back to string
    # ---------------------------
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded text:", decoded_text)


if __name__ == "__main__":
    main()
