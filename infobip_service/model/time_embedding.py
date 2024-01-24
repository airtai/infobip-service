import math
from typing import Any, Callable, Dict

import torch
from torch.nn import Linear

PERIODS_SECONDS = {
    "day": 24 * 60 * 60,
    "week": 7 * 24 * 60 * 60,
    "month": 365 * 24 * 60 * 60 / 12,
    "year": 365 * 24 * 60 * 60,
}

PERIODS = list(PERIODS_SECONDS.keys())


def expand_time_feature(x: torch.Tensor, period: str) -> float:
    return x % PERIODS_SECONDS[period]


def expand_time_features(d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    dd = {
        f"{name}_{period}": expand_time_feature(x, period)
        for name, x in d.items()
        for period in PERIODS
    }
    return {**dd}


def create_period_normalization(name: str, period: str) -> Callable[[Any], Any]:
    def _inner(
        x: Any,
        mean: float = PERIODS_SECONDS[period] / 2,
        std: float = PERIODS_SECONDS[period] / math.pi,
        sin: Any = torch.sin,
        cos: Any = torch.cos,
    ) -> Any:
        y = x - mean
        y = y / std
        return torch.stack([sin(y), cos(y)], -1)

    return _inner


class TimeEmbedding(torch.nn.Module):
    """Time embedding layer."""

    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        output_dim: int,
    ):
        """Time embedding layer initialization method."""
        super().__init__()
        self.name = name
        self.mean = mean
        self.std = std
        self.period_normalizations = {
            f"{name}_{period}_normalization": create_period_normalization(name, period)
            for period in PERIODS
        }
        self.linear = Linear(in_features=len(PERIODS) * 2 + 1, out_features=output_dim)
        self.activation = torch.nn.ELU()

    def forward(self, x: Any) -> Any:
        periods = [
            self.period_normalizations[f"{k}_normalization"](v)
            for k, v in expand_time_features({self.name: x}).items()
        ]
        time = (x - self.mean) / self.std

        y = torch.cat(periods + [torch.unsqueeze(time, dim=-1)], dim=-1)  # noqa: RUF005

        y = self.linear(y)
        y = self.activation(y)
        return y


def build_embedding_layer_time(
    name: str,
    mean: float,
    std: float,
    output_dim: int,
) -> Callable[[Any], Any]:
    return TimeEmbedding(name, mean, std, output_dim)
