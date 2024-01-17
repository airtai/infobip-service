from infobip_service.preprocesing import *

import tempfile
from pathlib import Path
from random import randrange, choice, sample
from typing import List, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from infobip_service.download import raw_data_path



def test_sample_time_map_df():
    raw = pd.DataFrame({
            "AccountId": [12345, 12345, 12345, 12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-10 13:27:01.246912",
                "2023-07-12 13:27:02.370368",
                "2023-07-12 13:27:03.493824",
                "2023-07-12 13:27:04.617280",
                "2023-07-10 13:27:05.740736",
            ],
            "DefinitionId": ["one", "one", "one", "two", "two", "three"],
            "ApplicationId": [None, None, None, None, None, None],
        }, index=pd.Index([1, 2, 2, 3, 3, 3], name="PersonId"))
    raw["OccurredTime"] = pd.to_datetime(raw["OccurredTime"])
    raw["DefinitionId"] = raw["DefinitionId"].astype("string[pyarrow]")
    raw["ApplicationId"] = raw["ApplicationId"].astype("string[pyarrow]")

    expected = pd.DataFrame({
            "AccountId": [12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-10 13:27:01.246912",
                "2023-07-10 13:27:05.740736",
            ],
            "DefinitionId": ["one", "one", "three"],
            "ApplicationId": [None, None, None],
        }, index=pd.Index([1, 2, 3], name="PersonId"))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    pd.testing.assert_frame_equal(_sample_time_map(raw, time_treshold=datetime(2023, 7, 11)), expected)

def test_sample_time_map_ddf():
    raw = pd.DataFrame({
            "AccountId": [12345, 12345, 12345, 12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-10 13:27:01.246912",
                "2023-07-12 13:27:02.370368",
                "2023-07-12 13:27:03.493824",
                "2023-07-12 13:27:04.617280",
                "2023-07-10 13:27:05.740736",
            ],
            "DefinitionId": ["one", "one", "one", "two", "two", "three"],
            "ApplicationId": [None, None, None, None, None, None],
        }, index=pd.Index([1, 2, 2, 3, 3, 3], name="PersonId"))
    raw["OccurredTime"] = pd.to_datetime(raw["OccurredTime"])
    raw["DefinitionId"] = raw["DefinitionId"].astype("string[pyarrow]")
    raw["ApplicationId"] = raw["ApplicationId"].astype("string[pyarrow]")

    expected = pd.DataFrame({
            "AccountId": [12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-10 13:27:01.246912",
                "2023-07-10 13:27:05.740736",
            ],
            "DefinitionId": ["one", "one", "three"],
            "ApplicationId": [None, None, None],
        }, index=pd.Index([1, 2, 3], name="PersonId"))

    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    actual = sample_time_map(dd.from_pandas(raw, npartitions=2), time_treshold=datetime(2023, 7, 11)).compute()
    pd.testing.assert_frame_equal(actual, expected)

def test_remove_without_history_df():
    df = pd.DataFrame({
        'OccurredTime': ['2022-01-01 10:00:00', '2022-01-01 11:00:00', '2022-01-01 12:00:00'],
        'Data': [1, 2, 3]
    })
    df["OccurredTime"] = pd.to_datetime(df["OccurredTime"])

    time_threshold = datetime.strptime('2022-01-01 11:30:00', '%Y-%m-%d %H:%M:%S')

    result = _remove_without_history(df, time_treshold=time_threshold)

    expected = pd.DataFrame({
        'OccurredTime': ['2022-01-01 10:00:00', '2022-01-01 11:00:00'],
        'Data': [1, 2]
    }, index=[0, 1])
    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])

    assert result.equals(expected)

def test_remove_without_history_ddf():
    # TODO: test
    pass

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

def test_get_next_event_df():
    user_history = pd.DataFrame({
            "AccountId": [12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-12 13:27:01.246912",
                "2023-07-28 13:27:05.740736",
            ],
            "DefinitionId": ["one", "one", "one"],
            "ApplicationId": [None, None, None],
        }, index=pd.Index([1, 1, 1], name="PersonId"))
    user_history["OccurredTime"] = pd.to_datetime(user_history["OccurredTime"])
    
    # TODO: transform to asserts
    _get_next_event(user_history, t0=datetime(2023, 7, 19))
    _get_next_event(user_history, t0=datetime(2023, 7, 29))

def test_get_next_event():
    get_next_event(user_history, t0=datetime(2023, 7, 19))
    get_next_event(user_history, t0=datetime(2023, 7, 29))
## History construction
user_history = pd.DataFrame({
        "AccountId": [12345, 12345, 12345],
        "OccurredTime": [
            "2023-07-10 13:27:00.123456",
            "2023-07-12 13:27:01.246912",
            "2023-07-28 13:27:05.740736",
        ],
        "DefinitionId": ["one", "one", "one"],
        "ApplicationId": [None, None, None],
    }, index=pd.Index([1, 1, 1], name="PersonId"))
user_history["OccurredTime"] = pd.to_datetime(user_history["OccurredTime"])
#| export
def pad_left(l: list, *, size: int) -> list:
    return np.pad(l, pad_width=(size-len(l),0), mode='empty')


def _create_user_history(user_history: pd.DataFrame, *, t0: datetime, history_size: int) -> pd.DataFrame:
    user_history = user_history[user_history['OccurredTime'] < t0]
    user_history = user_history.sort_values(by='OccurredTime', ascending=False)
    user_history = user_history.head(history_size)
    user_history = user_history.sort_values(by='OccurredTime')

    reconstructed_history = pd.DataFrame({
        "AccountId": pad_left(user_history["AccountId"].values, size=history_size),
        "DefinitionId": pad_left(user_history["DefinitionId"].values, size=history_size),
        "ApplicationId": pad_left(user_history["ApplicationId"].values, size=history_size),
        "OccurredTime": pad_left(user_history["OccurredTime"].values, size=history_size),
    }, index=pd.Index([f"h_{i}" for i in range(history_size)], name="HistoryId")).T

    return reconstructed_history
_create_user_history(user_history, t0=datetime(2023, 7, 13), history_size=3)
user_histories = pd.DataFrame({
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
    }, index=pd.Index([1, 1, 1, 2, 2], name="PersonId"))
user_histories["OccurredTime"] = pd.to_datetime(user_histories["OccurredTime"])
user_histories["AccountId"] = user_histories["AccountId"].astype("string[pyarrow]")

user_histories
# | export

def create_user_histories(user_histories: pd.DataFrame, *, t0: datetime, history_size: int) -> pd.DataFrame:
        try:
                return user_histories.groupby("PersonId").apply(lambda x: _create_user_history(x, t0=t0, history_size=history_size))
        except KeyError:
                print(user_histories)
create_user_histories(user_histories, t0=datetime(2023, 7, 16), history_size=2)
user_history = pd.DataFrame({
        "AccountId": [12345],
        "OccurredTime": [
            "2023-07-10 13:27:00.123456",
        ],
        "DefinitionId": ["one"],
        "ApplicationId": [None],
    }, index=pd.Index([1], name="PersonId"))
user_history["OccurredTime"] = pd.to_datetime(user_history["OccurredTime"])

create_user_histories(user_histories, t0=datetime(2023, 7, 16), history_size=2)
## Dataset sampling
def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)
assert datetime(2023, 7, 10) < random_date(datetime(2023, 7, 10), datetime(2023, 7, 28)) < datetime(2023, 7, 28)
def convert_datetime(time: str) -> datetime:
    try:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
def sample_user_histories(user_histories: pd.DataFrame, *, min_time: datetime, max_time: datetime, history_size: int) -> pd.DataFrame:    
    num_samples_to_go = len(user_histories) // history_size + 1


    max_time = min(user_histories["OccurredTime"].describe()["max"] + timedelta(days=1), max_time)
    min_time = max(user_histories["OccurredTime"].describe()["min"], min_time)

    user_histories_sample = None

    timestamp_misses = 0

    while num_samples_to_go > 0:
        t0 = random_date(min_time, max_time)
        filtered_index = user_histories[user_histories['OccurredTime'] < t0].index.unique()
        if len(filtered_index) == 0:
            timestamp_misses += 1
            continue
        
        chosen_user_history = user_histories[user_histories.index.isin([choice(filtered_index)])]
        
        next_event_timedelta = get_next_event(chosen_user_history, t0=t0)
        reconstructed_history = create_user_histories(chosen_user_history, t0=t0, history_size=history_size)
        reconstructed_history = pd.merge(reconstructed_history, next_event_timedelta, left_index=True, right_index=True)
        reconstructed_history.index = reconstructed_history.index.map(lambda x: (f'{x[0]}_{t0}', x[1]))

        if user_histories_sample is None:
            user_histories_sample = reconstructed_history
        else:
            user_histories_sample = pd.concat([user_histories_sample, reconstructed_history])
        
        num_samples_to_go -= 1
    
    print(f"Timestamp misses: {timestamp_misses}, missed ratio: {(timestamp_misses / (len(user_histories)+timestamp_misses))*100:.2f}%")

    return user_histories_sample
# može se patchat random date ovdje za test

user_histories_sample = sample_user_histories(user_histories, min_time=datetime(2023, 7, 9) , max_time=datetime(2023, 7, 20), history_size=2)
user_histories_sample
user_histories_sample = sample_user_histories(user_history, min_time=datetime(2023, 7, 9) , max_time=datetime(2023, 7, 20), history_size=2)
user_histories_sample
def prepare_data(ddf: dd.DataFrame, *, min_time: datetime, max_time: datetime, history_size: int) -> pd.DataFrame:
    meta = sample_user_histories(ddf.head(2, npartitions=-1), min_time=min_time, max_time=max_time, history_size=history_size)    

    sampled_data = ddf.map_partitions(sample_user_histories, min_time=min_time, max_time=max_time, history_size=history_size, meta=meta)

    return sampled_data
user_histories = pd.DataFrame({
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

user_histories["OccurredTime"] = pd.to_datetime(user_histories["OccurredTime"])
user_histories["AccountId"] = user_histories["AccountId"].astype("string[pyarrow]")

user_histories_ddf = dd.from_pandas(user_histories, npartitions=2)
user_histories_ddf
user_index = user_histories_ddf.index.unique().compute()
train_index = np.random.choice(user_index, size=1, replace=False)
validation_index = np.setdiff1d(user_index, train_index)

max_time = datetime.strptime(user_histories_ddf["OccurredTime"].describe().compute()["max"], "%Y-%m-%d %H:%M:%S.%f")
min_time = datetime.strptime(user_histories_ddf["OccurredTime"].describe().compute()["min"], "%Y-%m-%d %H:%M:%S.%f")

sampled_data = prepare_data(user_histories_ddf, history_size=8, min_time=min_time, max_time=max_time)

display(sampled_data.head(4, npartitions=-1))
user_histories = pd.DataFrame({
        "AccountId": [12345],
        "OccurredTime": [
            "2023-07-10 13:27:00.123456",
        ],
        "DefinitionId": ["two"],
        "ApplicationId": [None],
    }, index=pd.Index([1], name="PersonId"))

user_histories["OccurredTime"] = pd.to_datetime(user_histories["OccurredTime"])
user_histories["AccountId"] = user_histories["AccountId"].astype("string[pyarrow]")

user_histories_ddf = dd.from_pandas(user_histories, npartitions=2)
user_histories_ddf

user_index = user_histories_ddf.index.unique().compute()
train_index = np.random.choice(user_index, size=1, replace=False)
validation_index = np.setdiff1d(user_index, train_index)

max_time = datetime.strptime(user_histories_ddf["OccurredTime"].describe().compute()["max"], "%Y-%m-%d %H:%M:%S.%f") + timedelta(days=1)
min_time = datetime.strptime(user_histories_ddf["OccurredTime"].describe().compute()["min"], "%Y-%m-%d %H:%M:%S.%f") - timedelta(days=1)

sampled_data = prepare_data(user_histories_ddf, history_size=8, min_time=min_time, max_time=max_time)

display(sampled_data.head(4, npartitions=-1))