# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from models.embeddings.base import BaseEmbedding


class PositionalEmbedding(BaseEmbedding):
    def __init__(self, max_length: int, embedding_dim: int):
        """
        Positional embedding layer.

        Args:
            max_length (int): Maximum sequence length
            embedding_dim (int): Dimension of each positional embedding vector (D)
        """
        super().__init__()
        # Create a positional embedding matrix of shape (max_length, D)
        # Each row corresponds to a position in the sequence
        self.position_embeddings = nn.Embedding(max_length, embedding_dim)

        # Optional: small random initialization to start training
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get positional embeddings.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, sequence_length)

        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, seq_length = input_ids.shape

        # Create position indices for the sequence: shape (1, sequence_length), then expand to (batch_size,
        # sequence_length)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)

        # Lookup the positional embeddings
        return self.position_embeddings(positions)
