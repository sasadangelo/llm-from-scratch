# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
from tokenizer import SimpleTokenizer
from datasets import BookSource
from torch.utils.data import DataLoader
from datasets import LanguageModelingDataset

# --- Load book ---
# --- Configure pages to skip ---
# Pages are 1-based, ranges are inclusive
skip_pages = "1-14,79-80,91-92,149-150,253-254,416-418"
# This will load the PDF and handle skipped pages
book = BookSource(path="robinson-crusoe.pdf", skip_pages=skip_pages)
# BookSource.load() returns the full text as a single string
text = book.load()
print("Text length (chars):", len(text))
print("Preview:\n", text[:500])
print("-" * 60)

# --- Initialize the tokenizer ---
tokenizer = SimpleTokenizer()
tokenizer.train(text=text)
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Special tokens:", tokenizer.get_special_tokens())
print("-" * 60)

# --- Create language modeling dataset ---
context_length = 5
stride = 1
dataset = LanguageModelingDataset(text=text, tokenizer=tokenizer, context_length=context_length, stride=stride)

print("Number of samples in dataset:", len(dataset))

# --- Create DataLoader for batch training ---
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
    print("Batch: ", batch_idx)
    print()
    for i in range(input_ids.shape[0]):  # input_ids.shape[0] = batch_size
        input_text = tokenizer.decode(input_ids[i].tolist())
        target_text = tokenizer.decode(target_ids[i].tolist())
        print(f"    Sample {i}.")
        print("        input IDs: ", input_ids[i].tolist(), " -> target IDs: ", target_ids[i].tolist())
        print("        input text:", input_text, " -> target text: ", target_text)
    print("-" * 40)
    if batch_idx == 3:
        break  # just show first 3 batches
