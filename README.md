# LLM From Scratch

## Introduction
This project aims to build a **Language Model (LLM) from scratch** in an educational setting, to understand how tokenization, embeddings, and language model training work.
It is not intended for production use, but rather as a learning laboratory to experiment with NLP and deep learning concepts.

The project is mainly written in **Python**, with a **Rust crate** integrated for fast Byte Pair Encoding (BPE) tokenization.
The project structure follows a [Python Blueprint](https://github.com/sasadangelo/python-blueprint), providing a clean and modular code organization.

---

## Setup

### Prerequisites
- Python >= 3.10
- Rust + Cargo (for compiling the Rust module)
- `uv` task runner / development environment

### Installation and Setup

1. Synchronize development dependencies:

```bash
uv sync --dev
```

2. Build and install the Rust tokenizer module:

```bash
uv run maturin develop
```

3. Run the test suite:

```bash
uv run pytest -v
```

4. Run the main Python script:

```bash
uv run src/llmain.py
```

### Project Structure

```
src/ – main Python source code
src/tokenizer/rustbpe/ – Rust BPE tokenizer module
llmtest/ – unit tests
README.md – this file
```

### Notes

The Rust tokenizer provides fast encoding/decoding for text.
The project is intended as an educational tool to explore LLM internals.

