from typing import Dict

import torch

from .category_embedding import build_embedding_layer_category
from .time_embedding import build_embedding_layer_time


class ChurnModel(torch.nn.Module):
    """Churn model."""

    def __init__(
        self,
        definition_id_vocab_size: int,
        time_normalization_params: Dict[str, float],
        embedding_dim: int = 10,
        churn_bucket_size: int = 6,
    ):
        """Churn model initialization method."""
        super().__init__()

        self.definition_id_embedding = build_embedding_layer_category(
            "DefinitionId", definition_id_vocab_size, embedding_dim=embedding_dim
        )

        self.time_embedding = build_embedding_layer_time(
            "Time",
            mean=time_normalization_params["mean"],
            std=time_normalization_params["std"],
            output_dim=embedding_dim,
        )

        self.conv = torch.nn.Conv1d(2 * embedding_dim, 2 * embedding_dim, 2, stride=2)

        self.linear1 = torch.nn.Linear(2 * embedding_dim, 32)
        self.activation = torch.nn.ELU()

        self.linear2 = torch.nn.Linear(32, churn_bucket_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        definiton_embeddings = self.definition_id_embedding(x.select(-1, 0))
        time_embeddings = self.time_embedding(x.select(-1, 1))

        y = torch.cat([definiton_embeddings, time_embeddings], dim=-1)

        permutes = list(range(len(y.shape)))
        permutes[-1], permutes[-2] = permutes[-2], permutes[-1]

        y = y.permute(*permutes)

        while y.shape[-1] > 1:
            y = self.conv(y)

        y = y.permute(*permutes)

        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.softmax(y)

        return y
