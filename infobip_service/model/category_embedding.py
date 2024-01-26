from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Embedding


class ClassEmbedding(torch.nn.Module):
    """Class embedding layer."""

    def __init__(self, name: str, voc_size: int, embedding_dim: int):
        """Class embedding layer initialization method."""
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=voc_size,
            embedding_dim=embedding_dim,
            # name=f"{name}_embedding",
        )

    def forward(self, x: Any) -> Any:
        y = self.embedding(x)
        return y


def build_embedding_layer_category(
    name: str, voc_size: int, embedding_dim: int
) -> Callable[[Any], Any]:
    return ClassEmbedding(name, voc_size, embedding_dim)
