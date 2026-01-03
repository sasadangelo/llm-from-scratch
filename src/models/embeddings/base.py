# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEmbedding(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: torch.Tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        pass
