from typing import Any, Callable

from torch.nn import Embedding


def build_embedding_layer_category(
    name: str, voc_size: int, embedding_dim: int
) -> Callable[[Any], Any]:
    embedding = Embedding(
        num_embeddings=voc_size,
        embedding_dim=embedding_dim,
        # name=f"{name}_embedding",
    )

    def _inner(x: Any, embedding: Any = embedding) -> Any:
        y = embedding(x)
        return y

    return _inner
