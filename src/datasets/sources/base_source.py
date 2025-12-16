# -----------------------------------------------------------------------------
# Copyright (c) 2025 Salvatore D'Angelo, Code4Projects
# Licensed under the MIT License. See LICENSE.md for details.
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod


class BaseSource(ABC):
    @abstractmethod
    def load(self) -> str:
        """Load raw text from the source"""
        pass
