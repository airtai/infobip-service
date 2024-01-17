import tempfile
from pathlib import Path
from random import randrange, choice, sample
from typing import List, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

processed_data_path = Path() / '..' / 'data' / 'processed'

# Dataset preparation

def write_and_read_parquet(ddf: dd.DataFrame, *, path: Path, **kwargs) -> dd.DataFrame:
    ddf.to_parquet(path, **kwargs)
    return ddf.read_parquet(path, calculate_divisions=True)

def _sample_time_map(df: pd.DataFrame, *, time_treshold: datetime) -> pd.DataFrame:
    df = df[df['OccurredTime'] < time_treshold]
    return df

def sample_time_map(ddf: dd.DataFrame, *, time_treshold: datetime) -> dd.DataFrame:
    meta = _sample_time_map(ddf._meta, time_treshold=time_treshold)
    return ddf.map_partitions(_sample_time_map, time_treshold=time_treshold, meta=meta)

def _remove_without_history(df: pd.DataFrame, *, time_treshold: datetime) -> pd.DataFrame:
    indexes_with_history = df[df['OccurredTime'] < time_treshold].index
    return df[df.index.isin(indexes_with_history)]

def remove_without_history(ddf: dd.DataFrame, *, time_treshold: datetime) -> dd.DataFrame:
    meta = _remove_without_history(ddf._meta, time_treshold=time_treshold)
    return ddf.map_partitions(_remove_without_history, time_treshold=time_treshold, meta=meta)


## Train/test split

def split_data(ddf: dd.DataFrame, *, split_ratio: float = 0.8) -> (dd.DataFrame, dd.DataFrame):
    # ddf = ddf.set_index(ddf.index, sorted=True)
    user_index = ddf.index.unique().compute()
    train_index = np.random.choice(user_index, size=int(len(user_index)*split_ratio), replace=False)
    validation_index = np.setdiff1d(user_index, train_index)

    train_data = ddf.loc[train_index]
    validation_data = ddf.loc[validation_index]

    return train_data, validation_data

# Sample construction

## Next event

def _get_next_event(df: pd.DataFrame, *, t0: datetime) -> pd.DataFrame:
    _df = df[df['OccurredTime'] >= t0]
    return pd.DataFrame({
        "AccountId": [_df["AccountId"].values[0] if len(_df) > 0 else None],
        "DefinitionId": [_df["DefinitionId"].values[0] if len(_df) > 0 else None],
        "ApplicationId": [_df["ApplicationId"].values[0] if len(_df) > 0 else None],
        "OccurredTime": [_df["OccurredTime"].values[0] if len(_df) > 0 else None],
    }, index=pd.Index(['NextEvent'])).T

def get_next_event(df: pd.DataFrame, *, t0: datetime) -> pd.DataFrame:
    return df.groupby("PersonId").apply(lambda x: _get_next_event(x, t0=t0))

## History construction

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

def create_user_histories(user_histories: pd.DataFrame, *, t0: datetime, history_size: int) -> pd.DataFrame:
        try:
                return user_histories.groupby("PersonId").apply(lambda x: _create_user_history(x, t0=t0, history_size=history_size))
        except KeyError:
                print(user_histories)

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

def prepare_data(ddf: dd.DataFrame, *, min_time: datetime, max_time: datetime, history_size: int) -> pd.DataFrame:
    meta = sample_user_histories(ddf.head(2, npartitions=-1), min_time=min_time, max_time=max_time, history_size=history_size)    

    sampled_data = ddf.map_partitions(sample_user_histories, min_time=min_time, max_time=max_time, history_size=history_size, meta=meta)

    return sampled_data

def prepare_ddf(ddf: dd.DataFrame, *, history_size: int) -> dd.DataFrame:
    max_time = convert_datetime(ddf["OccurredTime"].describe().compute()["max"]) - timedelta(days=28)
    min_time = convert_datetime(ddf["OccurredTime"].describe().compute()["min"])

    sampled_data = prepare_data(ddf, history_size=history_size, min_time=min_time, max_time=max_time)

    return sampled_data