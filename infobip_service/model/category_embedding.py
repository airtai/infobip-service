from typing import Any, Callable, List, Union

import torch
from torch.nn import Embedding


def build_embedding_layer_category(
    name: str, voc: List[Union[str, int]], embedding_dim: int
) -> Callable[[Any], Any]:
    embedding = Embedding(
        num_embeddings=len(voc) + 1,
        embedding_dim=embedding_dim,
        # name=f"{name}_embedding",
    )

    def _lookup(x: str, voc: List[Any] = voc) -> int:
        try:
            return voc.index(x)
        except ValueError:
            return len(voc)

    def _inner(x: Any, embedding: Any = embedding) -> Any:
        y = torch.as_tensor([_lookup(x) for x in x])
        y = embedding(y)
        return y

    return _inner
