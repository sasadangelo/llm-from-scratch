# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from models.embeddings.base import BaseEmbedding


class TokenEmbedding(BaseEmbedding):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Token embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary (V)
            embedding_dim (int): Dimension of each token embedding vector (D)
        """
        super().__init__()
        # Create the token embedding matrix of shape (V, D)
        # Each row corresponds to a token vector
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional: small random initialization to start training
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get token embeddings.

        Args:
            token_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, sequence_length)

        Returns:
            torch.Tensor: Token embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.token_embeddings(token_ids)
