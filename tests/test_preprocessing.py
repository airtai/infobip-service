import tempfile
from pathlib import Path
from random import randrange, choice, sample
from typing import List, Any
from datetime import datetime, timedelta

import pytest
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from infobip_service.download import raw_data_path
from infobip_service.preprocessing import _sample_time_map, sample_time_map, _remove_without_history, remove_without_history, split_data, _get_next_event, get_next_event, create_user_histories, sample_user_histories, prepare_data, write_and_read_parquet, random_date, convert_datetime


user_histories_df = pd.DataFrame({
    "AccountId": [12345, 12345, 12345, 12345, 12345],
    "OccurredTime": [
        "2023-07-10 13:27:00.123456",
        "2023-07-12 13:27:01.246912",
        "2023-07-28 13:27:05.740736",
        "2023-07-13 13:27:01.246912",
        "2023-07-26 13:27:05.740736",
    ],
    "DefinitionId": ["one", "one", "one", "two", "two"],
    "ApplicationId": [None, None, None, None, None],
}, index=pd.Index([1, 1, 2, 2, 2], name="PersonId"))


user_histories_df["OccurredTime"] = pd.to_datetime(user_histories_df["OccurredTime"])
user_histories_df["DefinitionId"] = user_histories_df["DefinitionId"].astype("string[pyarrow]")
user_histories_df["ApplicationId"] = user_histories_df["ApplicationId"].astype("string[pyarrow]")

user_histories_ddf = dd.from_pandas(user_histories_df, npartitions=2)

user_history_df = pd.DataFrame({
        "AccountId": [12345],
        "OccurredTime": [
            "2023-07-28 13:27:05.740736",
        ],
        "DefinitionId": ["one"],
        "ApplicationId": [ None],
    }, index=pd.Index([1], name="PersonId"))

user_history_df["OccurredTime"] = pd.to_datetime(user_history_df["OccurredTime"])
user_history_df["DefinitionId"] = user_history_df["DefinitionId"].astype("string[pyarrow]")
user_history_df["ApplicationId"] = user_history_df["ApplicationId"].astype("string[pyarrow]")

user_history_ddf = dd.from_pandas(user_history_df, npartitions=1)


def test_sample_time_map_df():
    expected = pd.DataFrame({
            "AccountId": [12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-12 13:27:01.246912",
                "2023-07-13 13:27:01.246912",
            ],
            "DefinitionId": ["one", "one", "two"],
            "ApplicationId": [None, None, None],
        }, index=pd.Index([1, 1, 2], name="PersonId"))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(_sample_time_map(user_histories_df, time_treshold=datetime(2023, 7, 14)), expected)

def test_sample_time_map_ddf():
    expected = pd.DataFrame({
            "AccountId": [12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-12 13:27:01.246912",
                "2023-07-13 13:27:01.246912",
            ],
            "DefinitionId": ["one", "one", "two"],
            "ApplicationId": [None, None, None],
        }, index=pd.Index([1, 1, 2], name="PersonId"))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(sample_time_map(user_histories_ddf, time_treshold=datetime(2023, 7, 14)).compute(), expected)

def test_remove_without_history_df():
    result = _remove_without_history(user_histories_df, time_treshold=datetime(2023, 7, 12))

    expected = pd.DataFrame({
        "AccountId": [12345, 12345],
        "OccurredTime": [
            "2023-07-10 13:27:00.123456",
            "2023-07-12 13:27:01.246912",
        ],
        "DefinitionId": ["one", "one"],
        "ApplicationId": [None, None],
    }, index=pd.Index([1, 1], name="PersonId"))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(result, expected)

def test_remove_without_history_ddf():
    result = remove_without_history(user_histories_ddf, time_treshold=datetime(2023, 7, 12)).compute()

    expected = pd.DataFrame({
        "AccountId": [12345, 12345],
        "OccurredTime": [
            "2023-07-10 13:27:00.123456",
            "2023-07-12 13:27:01.246912",
        ],
        "DefinitionId": ["one", "one"],
        "ApplicationId": [None, None],
    }, index=pd.Index([1, 1], name="PersonId"))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(result, expected)

def test_split_data():
    user_histories = pd.DataFrame({
            "AccountId": [12345, 12345, 12345, 12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-12 13:27:01.246912",
                "2023-07-28 13:27:05.740736",
                "2023-07-13 13:27:01.246912",
                "2023-07-26 13:27:05.740736",
                "2023-07-26 13:27:05.740736",

            ],
            "DefinitionId": ["one", "one", "one", "two", "two", "three"],
            "ApplicationId": [None, None, None, None, None, None],
        }, index=pd.Index([1, 1, 2, 2, 3, 3], name="PersonId"))

    user_histories["OccurredTime"] = pd.to_datetime(user_histories["OccurredTime"])
    user_histories["AccountId"] = user_histories["AccountId"].astype("string[pyarrow]")

    user_histories_ddf = dd.from_pandas(user_histories, npartitions=5)
    user_histories_ddf

    with tempfile.TemporaryDirectory() as tmpdirname:
        train, validation = split_data(user_histories_ddf, split_ratio=0.7)

        train = write_and_read_parquet(train, path = tmpdirname + "/train")
        validation = write_and_read_parquet(validation, path = tmpdirname + "/validation")

        assert train.npartitions == 2
        assert validation.npartitions == 1

        assert len(train.index.unique().compute()) == 2
        assert len(validation.index.unique().compute()) == 1

def test_get_next_event_single():
    expected = pd.DataFrame({
        "AccountId": [12345],
        "DefinitionId": ["one"],
        "ApplicationId": [None],
        "OccurredTime": ["2023-07-28 13:27:05.740736"],
    }, index=pd.Index(['NextEvent']))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(_get_next_event(user_history_df, t0=datetime(2023, 7, 19)), expected.T)

def test_get_next_event_single_empty():
    expected = pd.DataFrame({
        "AccountId": [None],
        "DefinitionId": [None],
        "ApplicationId": [None],
        "OccurredTime": [None],
    }, index=pd.Index(['NextEvent']))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(_get_next_event(user_history_df, t0=datetime(2023, 7, 29)), expected.T)

@pytest.mark.skip("TODO: figure out multiindex construction")
def test_get_next_event_group_by():
    expected = pd.DataFrame({
        "AccountId": [None, 12345],
        "DefinitionId": [None, "one"],
        "ApplicationId": [None, None],
        "OccurredTime": [None, "2023-07-28 13:27:05.740736"],
        "PersonId": [1, 2]
    }, index=pd.Index(['NextEvent']*2))

    print(get_next_event(user_histories_df, t0=datetime(2023, 7, 19)))


def test_create_user_histories():
    # TODO: add asserts
    create_user_histories(user_histories_df, t0=datetime(2023, 7, 16), history_size=2)

def test_create_user_histories_single():
    # TODO: add asserts
    create_user_histories(user_history_df, t0=datetime(2023, 7, 16), history_size=2)


def test_random_date():
    assert datetime(2023, 7, 10) < random_date(datetime(2023, 7, 10), datetime(2023, 7, 28)) < datetime(2023, 7, 28)

def test_sample_user_histories():
    # TODO: patch random here and add asserts
    user_histories_sample = sample_user_histories(user_histories_df, min_time=datetime(2023, 7, 9) , max_time=datetime(2023, 7, 20), history_size=2)
    print(user_histories_sample)

def test_sample_user_histories_single():
    # TODO: patch random here and add asserts
    user_histories_sample = sample_user_histories(user_history_df, min_time=datetime(2023, 7, 9) , max_time=datetime(2023, 8, 20), history_size=2)
    print(user_histories_sample)

def test_prepare_data_ddf():
    # TODO: asserts
    max_time = convert_datetime(user_histories_ddf["OccurredTime"].describe().compute()["max"])
    min_time = convert_datetime(user_histories_ddf["OccurredTime"].describe().compute()["min"])

    sampled_data = prepare_data(user_histories_ddf, history_size=8, min_time=min_time, max_time=max_time)

    print(sampled_data.head(4, npartitions=-1))

def test_prepare_data_ddf_single():
    # TODO: asserts
    max_time = convert_datetime(user_history_ddf["OccurredTime"].describe().compute()["max"]) + timedelta(days=1)
    min_time = convert_datetime(user_history_ddf["OccurredTime"].describe().compute()["min"])

    sampled_data = prepare_data(user_history_ddf, history_size=8, min_time=min_time, max_time=max_time)

    print(sampled_data.head(4, npartitions=-1))
