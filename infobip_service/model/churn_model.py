import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

import torch
from scipy import interpolate

from infobip_service.dataset.load_dataset import bins

from .category_embedding import build_embedding_layer_category
from .time_embedding import build_embedding_layer_time


class ChurnModel(torch.nn.Module):
    """Churn model."""

    def __init__(
        self,
        definition_id_vocab_size: int,
        time_normalization_params: dict[str, float],
        embedding_dim: int = 10,
        churn_bucket_size: int = 7,
        history_size: int = 32,
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

        self.convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(2 * embedding_dim, 2 * embedding_dim, 2, stride=2)
                for _ in range(int(math.sqrt(history_size)))
            ]
        )

        self.linear1 = torch.nn.Linear(2 * embedding_dim, 32)
        self.activation = torch.nn.ELU()

        self.linear2 = torch.nn.Linear(32, churn_bucket_size)

    def forward(self, x: torch.Tensor) -> Any:
        definiton_embeddings = self.definition_id_embedding(x.select(-1, 0))
        time_embeddings = self.time_embedding(x.select(-1, 1))

        y = torch.cat([definiton_embeddings, time_embeddings], dim=-1)

        permutes = list(range(len(y.shape)))
        permutes[-1], permutes[-2] = permutes[-2], permutes[-1]

        y = y.permute(*permutes)

        for conv in self.convolutions:
            y = conv(y)

        y = y.permute(*permutes)

        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)

        return y.squeeze(-2)


def interpolate_cdf_from_pdf(
    pdf: list[float], bins: list[int] = bins
) -> Callable[[float], float]:
    x = [*bins, bins[-1] * 1000]
    y = [0.0] + [sum(pdf[:i]) for i in range(1, len(pdf) + 1)]
    f = interpolate.interp1d(x, y)
    return f  # type: ignore


def cdf_after_x_seconds(
    cdf: Callable[[float], float], seconds: float
) -> Callable[[float], float]:
    prob = cdf(seconds)
    coef = 1 / (1 - prob)
    bias = -coef * prob
    return lambda x: coef * cdf(x) + bias


def churn(
    pdf: list[float], seconds: float, time_to_churn: int, bins: list[int] = bins
) -> float:
    cdf = interpolate_cdf_from_pdf(pdf, bins=bins)
    cdf = cdf_after_x_seconds(cdf, seconds)
    return 1 - cdf(max(time_to_churn, seconds))


class ChurnProbabilityModel(torch.nn.Module):
    """Churn model."""

    def __init__(
        self,
        churn_model: ChurnModel,
        bins: list[int] = bins,
    ):
        """Churn model initialization method."""
        super().__init__()
        self.time_to_churn = bins[-1]
        self.churn_model = churn_model
        self.softmax = torch.nn.Softmax(dim=-1)
        self.bins = bins

    def forward(self, x: torch.Tensor, observed_time: datetime) -> Any:
        event_probabilities = self.softmax(self.churn_model(x).detach())
        last_events_times = x.select(-1, 1).select(-1, -1)

        observed_time = torch.tensor(observed_time.timestamp())
        seconds_from_last_events = (observed_time - last_events_times).float()

        combined = torch.cat(  # type: ignore
            [event_probabilities, seconds_from_last_events.unsqueeze(-1)], axis=-1
        )

        def calculate_churn_for_row(row: torch.Tensor) -> float:
            return churn(
                row[0:-1], row[-1], time_to_churn=self.time_to_churn, bins=self.bins
            )

        if len(combined.shape) == 1:
            return torch.tensor(calculate_churn_for_row(combined))  # type: ignore
        else:
            return torch.tensor([calculate_churn_for_row(row) for row in combined])  # type: ignore
