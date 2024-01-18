from datetime import datetime, timedelta
from pathlib import Path
from random import choice, randrange
from typing import Any, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from infobip_service.download import raw_data_path

processed_data_path = Path() / ".." / "data" / "processed"

# Dataset preparation


def write_and_read_parquet(
    ddf: dd.DataFrame,  # type: ignore
    *,
    path: Path,
    **kwargs: Any,
) -> dd.DataFrame:  # type: ignore
    ddf.to_parquet(path, **kwargs)
    return dd.read_parquet(path, calculate_divisions=True)  # type: ignore


def _sample_time_map(df: pd.DataFrame, *, time_treshold: datetime) -> pd.DataFrame:
    df = df[df["OccurredTime"] < time_treshold]
    return df


def sample_time_map(ddf: dd.DataFrame, *, time_treshold: datetime) -> dd.DataFrame:  # type: ignore
    meta = _sample_time_map(ddf._meta, time_treshold=time_treshold)
    return ddf.map_partitions(_sample_time_map, time_treshold=time_treshold, meta=meta)


def _remove_without_history(
    df: pd.DataFrame, *, time_treshold: datetime
) -> pd.DataFrame:
    indexes_with_history = df[df["OccurredTime"] < time_treshold].index
    return df[df.index.isin(indexes_with_history)]


def remove_without_history(
    ddf: dd.DataFrame,  # type: ignore
    *,
    time_treshold: datetime,
) -> dd.DataFrame:  # type: ignore
    meta = _remove_without_history(ddf._meta, time_treshold=time_treshold)
    return ddf.map_partitions(
        _remove_without_history, time_treshold=time_treshold, meta=meta
    )


## Train/test split


def split_data(
    ddf: dd.DataFrame,  # type: ignore
    *,
    split_ratio: float = 0.8,
) -> Tuple[dd.DataFrame, dd.DataFrame]:  # type: ignore
    # ddf = ddf.set_index(ddf.index, sorted=True)
    user_index = ddf.index.unique().compute()
    train_index = np.random.choice(
        user_index, size=int(len(user_index) * split_ratio), replace=False
    )
    validation_index = np.setdiff1d(user_index, train_index)

    train_data = ddf.loc[train_index]
    validation_data = ddf.loc[validation_index]

    return train_data, validation_data


# Sample construction

## Next event


def _get_next_event(df: pd.DataFrame, *, t0: datetime) -> pd.DataFrame:
    _df = df[df["OccurredTime"] >= t0]
    return pd.DataFrame(
        {
            "AccountId": [_df["AccountId"].values[0] if len(_df) > 0 else None],
            "DefinitionId": [_df["DefinitionId"].values[0] if len(_df) > 0 else None],
            "ApplicationId": [_df["ApplicationId"].values[0] if len(_df) > 0 else None],
            "OccurredTime": [_df["OccurredTime"].values[0] if len(_df) > 0 else None],
        },
        index=pd.Index(["NextEvent"]),
    ).T


def get_next_event(df: pd.DataFrame, *, t0: datetime) -> pd.DataFrame:
    return df.groupby("PersonId").apply(lambda x: _get_next_event(x, t0=t0))


## History construction


def pad_left(
    array_to_pad: np.ndarray[Any, np.dtype[Any]], *, size: int
) -> np.ndarray[Any, np.dtype[Any]]:
    return np.pad(array_to_pad, pad_width=(size - len(array_to_pad), 0), mode="empty")


def _create_user_history(
    user_history: pd.DataFrame, *, t0: datetime, history_size: int
) -> pd.DataFrame:
    user_history = user_history[user_history["OccurredTime"] < t0]
    user_history = user_history.sort_values(by="OccurredTime", ascending=False)
    user_history = user_history.head(history_size)
    user_history = user_history.sort_values(by="OccurredTime")

    reconstructed_history = pd.DataFrame(
        {
            "AccountId": pad_left(user_history["AccountId"].values, size=history_size),
            "DefinitionId": pad_left(
                user_history["DefinitionId"].values, size=history_size
            ),
            "ApplicationId": pad_left(
                user_history["ApplicationId"].values, size=history_size
            ),
            "OccurredTime": pad_left(
                user_history["OccurredTime"].values, size=history_size
            ),
        },
        index=pd.Index([f"h_{i}" for i in range(history_size)], name="HistoryId"),
    ).T

    return reconstructed_history


def create_user_histories(
    user_histories: pd.DataFrame, *, t0: datetime, history_size: int
) -> pd.DataFrame:
    return user_histories.groupby("PersonId").apply(
        lambda x: _create_user_history(x, t0=t0, history_size=history_size)
    )


## Dataset sampling


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)  # nosec
    return start + timedelta(seconds=random_second)


def convert_datetime(time: str) -> datetime:
    try:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")


def sample_user_histories(
    user_histories: pd.DataFrame,
    *,
    min_time: datetime,
    max_time: datetime,
    history_size: int,
) -> pd.DataFrame:
    num_samples_to_go = len(user_histories) // history_size + 1

    max_time = min(
        user_histories["OccurredTime"].describe()["max"] + timedelta(days=1), max_time
    )
    min_time = max(user_histories["OccurredTime"].describe()["min"], min_time)

    user_histories_sample = None

    while num_samples_to_go > 0:
        t0 = random_date(min_time, max_time)
        filtered_index = user_histories[
            user_histories["OccurredTime"] < t0
        ].index.unique()
        if len(filtered_index) == 0:
            continue

        chosen_user_history = user_histories[
            user_histories.index.isin([choice(filtered_index)])  # nosec
        ]

        next_event_timedelta = get_next_event(chosen_user_history, t0=t0)
        reconstructed_history = create_user_histories(
            chosen_user_history, t0=t0, history_size=history_size
        )
        reconstructed_history = pd.merge(
            reconstructed_history,
            next_event_timedelta,
            left_index=True,
            right_index=True,
        )
        reconstructed_history.index = reconstructed_history.index.map(
            lambda x, t0=t0: (f"{x[0]}_{t0}", x[1])
        )

        if user_histories_sample is None:
            user_histories_sample = reconstructed_history
        else:
            user_histories_sample = pd.concat(
                [user_histories_sample, reconstructed_history]
            )

        num_samples_to_go -= 1

    return user_histories_sample


def prepare_data(
    ddf: dd.DataFrame,  # type: ignore
    *,
    min_time: datetime,
    max_time: datetime,
    history_size: int,
) -> pd.DataFrame:
    meta = sample_user_histories(
        ddf.head(2, npartitions=-1),
        min_time=min_time,
        max_time=max_time,
        history_size=history_size,
    )

    sampled_data = ddf.map_partitions(
        sample_user_histories,
        min_time=min_time,
        max_time=max_time,
        history_size=history_size,
        meta=meta,
    )

    return sampled_data


def prepare_ddf(ddf: dd.DataFrame, *, history_size: int) -> dd.DataFrame:  # type: ignore
    max_time = convert_datetime(
        ddf["OccurredTime"].describe().compute()["max"]
    ) - timedelta(days=28)
    min_time = convert_datetime(ddf["OccurredTime"].describe().compute()["min"])

    sampled_data = prepare_data(
        ddf, history_size=history_size, min_time=min_time, max_time=max_time
    )

    return sampled_data


def preprocess_train_validation(
    raw_data_path: Path, preprocessed_data_path: Path
) -> Tuple[dd.DataFrame, dd.DataFrame]:  # type: ignore
    # Read raw data
    print("Reading raw data...")
    raw_data = dd.read_parquet(raw_data_path)  # type: ignore
    print("Raw data read.")

    # Calculate time threshold
    print("Calculating time threshold...")
    time_stats = raw_data["OccurredTime"].describe().compute()
    max_time = datetime.strptime(time_stats["max"], "%Y-%m-%d %H:%M:%S.%f")
    time_treshold = max_time - timedelta(days=28)
    print("Time threshold calculated.")

    # Time thresholding
    print("Time thresholding...")
    time_cutoff_data = sample_time_map(raw_data, time_treshold=time_treshold)
    time_cutoff_data = write_and_read_parquet(
        time_cutoff_data, path=processed_data_path / "time_cutoff_data.parquet"
    )
    print("Time thresholding done.")

    # Remove users without history
    print("Removing users without history...")
    data_before_horizon = remove_without_history(
        time_cutoff_data, time_treshold=time_treshold - timedelta(days=28)
    )
    data_before_horizon = write_and_read_parquet(
        data_before_horizon,
        path=processed_data_path / "data_before_horizon.parquet",
    )
    print("Users without history removed.")

    # Train/test split
    print("Splitting data...")
    train_raw, validation_raw = split_data(data_before_horizon, split_ratio=0.8)
    train_raw = write_and_read_parquet(
        train_raw.repartition(partition_size="10MB"),
        path=processed_data_path / "train_raw.parquet",
    )
    validation_raw = write_and_read_parquet(
        validation_raw.repartition(partition_size="10MB"),
        path=processed_data_path / "validation_raw.parquet",
    )
    print("Data split to train/validation.")

    # Prepare data
    print("Preparing data...")
    train_prepared = write_and_read_parquet(
        prepare_ddf(train_raw, history_size=64),
        path=processed_data_path / "train_prepared.parquet",
    )
    validation_prepared = write_and_read_parquet(
        prepare_ddf(validation_raw, history_size=64),
        path=processed_data_path / "validation_prepared.parquet",
    )
    print("Data prepared.")

    return train_prepared, validation_prepared


def sample_test_histories(
    user_histories: pd.DataFrame,
    *,
    horizon_time: datetime,
    history_size: int,
) -> pd.DataFrame:
    filtered_index = user_histories[
        user_histories["OccurredTime"] < horizon_time
        and user_histories["OccurredTime"] > horizon_time - timedelta(days=28)
    ].index.unique()

    user_histories_sample = None

    for index in filtered_index:
        chosen_user_history = user_histories[
            user_histories.index.isin([index])  # nosec
        ]
        next_event_timedelta = get_next_event(chosen_user_history, t0=horizon_time)
        reconstructed_history = create_user_histories(
            chosen_user_history, t0=horizon_time, history_size=history_size
        )
        reconstructed_history = pd.merge(
            reconstructed_history,
            next_event_timedelta,
            left_index=True,
            right_index=True,
        )

        if user_histories_sample is None:
            user_histories_sample = reconstructed_history
        else:
            user_histories_sample = pd.concat(
                [user_histories_sample, reconstructed_history]
            )

    return user_histories_sample


def prepare_test_data(
    ddf: dd.DataFrame,  # type: ignore
    *,
    horizon_time: datetime,
    history_size: int,
) -> pd.DataFrame:  # type: ignore
    meta = sample_test_histories(
        ddf.head(2, npartitions=-1),
        horizon_time=horizon_time,
        history_size=history_size,
    )

    sampled_data = ddf.map_partitions(
        sample_user_histories,
        horizon_time=horizon_time,
        history_size=history_size,
        meta=meta,
    )

    return sampled_data


def preprocess_test(raw_data_path: Path, processed_data_path: Path) -> dd.DataFrame:  # type: ignore
    # Read raw data
    print("Reading raw data...")
    raw_data = dd.read_parquet(raw_data_path)  # type: ignore
    print("Raw data read.")

    # Calculate time threshold
    print("Calculating max time...")
    time_stats = raw_data["OccurredTime"].describe().compute()
    max_time = datetime.strptime(time_stats["max"], "%Y-%m-%d %H:%M:%S.%f")
    print("Max time calculated.")

    print("Removing users without history...")
    data_before_horizon = remove_without_history(
        raw_data, time_treshold=max_time - timedelta(days=28)
    )
    data_before_horizon = write_and_read_parquet(
        data_before_horizon,
        path=processed_data_path / "test_data_before_horizon.parquet",
    )
    print("Users without history removed.")

    # Prepare data
    print("Preparing data...")
    data_prepared = prepare_test_data(
        data_before_horizon.repartition(partition_size="5MB"),
        history_size=64,
        horizon_time=max_time - timedelta(days=28),
    )
    data_prepared = write_and_read_parquet(
        data_prepared,
        path=processed_data_path / "test_prepared.parquet",
    )
    print("Data prepared.")

    return data_prepared


if __name__ == "__main__":
    cluster = LocalCluster()  # type: ignore
    client = Client(cluster)  # type: ignore

    print(client)

    try:
        preprocess_test(raw_data_path, processed_data_path)
        preprocess_train_validation(raw_data_path, processed_data_path)
    finally:
        client.close()  # type: ignore
        cluster.close()  # type: ignore