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


def build_embedding_layer_time(
    name: str,
    mean_and_std: Dict[str, float],
    output_dim: int,
) -> Callable[[Any], Any]:
    mean, std = mean_and_std["mean"], (mean_and_std["std"])

    period_normalizations = {
        f"{name}_{period}_normalization": create_period_normalization(name, period)
        for period in PERIODS
    }

    linear = Linear(in_features=len(PERIODS) * 2 + 1, out_features=output_dim)
    activation = torch.nn.ELU()

    def _inner(
        x: Any,
        mean: float = mean,
        std: float = std,
        period_normalizations: Any = period_normalizations,
        linear: Linear = linear,
    ) -> Any:
        periods = [
            period_normalizations[f"{k}_normalization"](v)
            for k, v in expand_time_features({name: x}).items()
        ]
        time = (x - mean) / std
        y = torch.cat(periods + [torch.unsqueeze(time, 0)])

        y = linear(y.T)
        y = activation(y)
        return y

    return _inner
