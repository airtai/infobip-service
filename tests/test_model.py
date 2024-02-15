import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from infobip_service.dataset.load_dataset import UserHistoryDataset
from infobip_service.dataset.preprocessing import processed_data_path
from infobip_service.model import ChurnModel, ChurnProbabilityModel
from infobip_service.model.churn_model import (
    churn,
    interpolate_cdf_from_pdf,
)


@pytest.mark.skip(reason="Dataset not available on CI/CD")
def test_model_forward():
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    with Path.open(processed_data_path / "time_stats.json", "rb") as f:
        time_stats = json.load(f)

    dataset = DataLoader(
        UserHistoryDataset(
            processed_data_path / "validation_prepared.parquet",
            definitionId_vocab=vocab,
        ),
        batch_size=4,
        pin_memory=True,
    )

    model = ChurnModel(
        definition_id_vocab_size=len(vocab) + 1,
        time_normalization_params=time_stats,
        embedding_dim=10,
        churn_bucket_size=6,
    )

    x, _ = next(iter(dataset))

    assert model(x).shape == (4, 1, 6)


def test_interpolate_cdf_from_pdf():
    pdf = [0.1, 0.2, 0.3, 0.4]
    buckets = [0, 1, 5, 28]

    cdf = interpolate_cdf_from_pdf(pdf, buckets)

    for i in np.linspace(1, 28, 100):
        assert 0 <= cdf(i) <= 1
        assert cdf(i - 1) <= cdf(i)


def test_churn():
    pdf = [0.1, 0.2, 0.3, 0.4]
    buckets = [0, 1, 5, 28]

    for i in np.linspace(1, 28, 100):
        assert 0 <= churn(pdf, seconds=i, time_to_churn=28, bins=buckets) <= 1
        assert churn(pdf, seconds=i - 1, time_to_churn=28, bins=buckets) <= churn(
            pdf, seconds=i, time_to_churn=28, bins=buckets
        )


@pytest.mark.skip(reason="Dataset not available on CI/CD")
def test_churn_probaility_model():
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    with Path.open(processed_data_path / "time_stats.json", "rb") as f:
        time_stats = json.load(f)

    dataset = DataLoader(
        UserHistoryDataset(
            processed_data_path / "validation_prepared.parquet",
            definitionId_vocab=vocab,
        ),
        batch_size=4,
        pin_memory=True,
    )

    model = ChurnModel(
        definition_id_vocab_size=len(vocab) + 1,
        time_normalization_params=time_stats,
        embedding_dim=10,
        churn_bucket_size=6,
    )
    model = ChurnProbabilityModel(model)

    x, _ = next(iter(dataset))

    assert model(x, datetime.now()).shape == torch.Size([4])
