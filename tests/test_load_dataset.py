import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from infobip_service.dataset.load_dataset import (
    UserHistoryDataset,
    _bin_timedelta,
    bin_next_event_user_history,
)
from infobip_service.dataset.preprocessing import processed_data_path

user_history = pd.DataFrame(
    {
        "AccountId": [12345, 12345, 12345],
        "OccurredTime": [
            "2023-07-10 13:27:00.123456",
            "2023-07-12 13:27:01.246912",
            "2023-07-28 13:27:05.740736",
        ],
        "DefinitionId": ["one", "one", "one"],
        "ApplicationId": [None, None, None],
    },
    index=pd.Index([1, 1, 1], name="PersonId"),
)
user_history["OccurredTime"] = pd.to_datetime(user_history["OccurredTime"])


def test_bin_timedelta():
    assert _bin_timedelta(timedelta(days=0)) == 0
    assert _bin_timedelta(timedelta(days=2)) == 1
    assert _bin_timedelta(timedelta(days=4)) == 2
    assert _bin_timedelta(timedelta(days=8)) == 3
    assert _bin_timedelta(timedelta(days=16)) == 4
    assert _bin_timedelta(timedelta(days=32)) == 5


def test_bin_next_event_user_history():
    assert (
        bin_next_event_user_history(
            datetime(2023, 7, 10, 23, 59), t0=datetime(2023, 7, 10)
        )
        == 0
    )  # 1 day to first event
    assert (
        bin_next_event_user_history(datetime(2023, 7, 11), t0=datetime(2023, 7, 10))
        == 1
    )  # 1.0000001 day to first event
    assert (
        bin_next_event_user_history(datetime(2023, 7, 13), t0=datetime(2023, 7, 10))
        == 2
    )  # 3 days to first event
    assert (
        bin_next_event_user_history(datetime(2023, 7, 17), t0=datetime(2023, 7, 10))
        == 3
    )  # 7 days to first event
    assert (
        bin_next_event_user_history(datetime(2023, 7, 25), t0=datetime(2023, 7, 10))
        == 4
    )  # 14 days to first event
    assert (
        bin_next_event_user_history(datetime(2023, 8, 7), t0=datetime(2023, 7, 10)) == 5
    )  # 28 days to first event
    assert (
        bin_next_event_user_history(datetime(2023, 8, 11), t0=datetime(2023, 7, 10))
        == 5
    )  # 32 days to first event
    assert (
        bin_next_event_user_history(None, t0=datetime(2023, 7, 10)) == 5
    )  # No next event


@pytest.mark.skip(reason="Dataset not available on CI/CD")
def test_train_dataset():
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    train_dataset = UserHistoryDataset(
        processed_data_path / "train_prepared.parquet", definitionId_vocab=vocab
    )

    for i in range(100):
        x, y = train_dataset[i]
        assert x.shape == (64, 2)
        assert y >= 0
        assert y <= 5


@pytest.mark.skip(reason="Dataset not available on CI/CD")
def test_test_dataset():
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    test_dataset = UserHistoryDataset(
        processed_data_path / "test_prepared.parquet", definitionId_vocab=vocab
    )

    for i in range(100):
        x, y = test_dataset[i]
        assert x.shape == (64, 2)
        assert y >= 0
        assert y <= 5


@pytest.mark.skip(reason="Dataset not available on CI/CD")
def test_val_dataset():
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    val_dataset = UserHistoryDataset(
        processed_data_path / "validation_prepared.parquet", definitionId_vocab=vocab
    )

    for i in range(100):
        x, y = val_dataset[i]
        assert x.shape == (64, 2)
        assert y >= 0
        assert y <= 5
