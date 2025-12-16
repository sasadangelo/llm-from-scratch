# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        vocab_size: dimensione del vocabolario (V)
        embedding_dim: dimensione dei vettori embedding (D)
        """
        super().__init__()
        # create the token embedding matrix of shape (V, D)
        # V=vocabulary size
        # D=vector dimension
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # optional: small random initialization
        # this helps to start training with non-zero embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: shape (batch_size, context_length)
        return: shape (batch_size, context_length, embedding_dim)
        """
        return self.token_embeddings(token_ids)
