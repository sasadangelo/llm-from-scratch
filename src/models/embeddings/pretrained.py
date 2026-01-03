# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
import torch
from transformers import AutoModel
from models.embeddings.base import BaseEmbedding


class PretrainedEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, freeze: bool = True):
        """
        Pretrained token embeddings compatible with TokenEmbedding.

        Args:
            model_name (str): HuggingFace model name (e.g., 'gpt2', 'bert-base-uncased')
            freeze (bool): If True, the pretrained model weights are frozen and not updated during training.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids (torch.Tensor): Token IDs, shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Embeddings, shape (batch_size, seq_length, embedding_dim)
        """
        outputs = self.model(input_ids=token_ids)
        return outputs.last_hidden_state
