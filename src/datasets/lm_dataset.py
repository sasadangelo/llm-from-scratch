# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
from tokenizer import BaseTokenizer


class LanguageModelingDataset(Dataset):
    def __init__(
        self,
        text: str,
        tokenizer: BaseTokenizer,
        context_length: int = 128,
        stride: int = 1,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride

        self.token_ids = tokenizer.encode(text)

        # Calculate the number of sequences (input/target pairs) that can be extracted from the text
        # given the context length and stride. The "-1" accounts for the target being shifted by one token
        # relative to the input.
        self.num_samples = (len(self.token_ids) - context_length - 1) // stride

    # Return the total number of samples in the dataset.
    # This is required by PyTorch's Dataset class to know the dataset size.
    def __len__(self):
        return self.num_samples

    # Return a single sample from the dataset at the given index `idx`.
    # It extracts a slice of token IDs of length `context_length` starting from `start`.
    # `input_ids` are the tokens used as input to the model.
    # `target_ids` are the same tokens shifted by one position, used as labels for next-token prediction.
    # Both are converted to PyTorch tensors of type long.
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length

        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1 : end + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )
