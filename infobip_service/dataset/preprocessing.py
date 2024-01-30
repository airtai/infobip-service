import json
from datetime import datetime, timedelta
from pathlib import Path
from random import choice, randrange
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from infobip_service.dataset.download import raw_data_path
from infobip_service.logger import get_logger, supress_timestamps

supress_timestamps(False)
logger = get_logger(__name__)

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
    df: pd.DataFrame,
    *,
    time_treshold: datetime,
    latest_event_delta: timedelta | None = None,
) -> pd.DataFrame:
    if latest_event_delta is not None:
        indexes_with_history = df[
            (df["OccurredTime"] < time_treshold)
            & (df["OccurredTime"] > time_treshold - latest_event_delta)
        ].index
    else:
        indexes_with_history = df[df["OccurredTime"] < time_treshold].index
    return df[df.index.isin(indexes_with_history)]


def remove_without_history(
    ddf: dd.DataFrame,  # type: ignore
    *,
    time_treshold: datetime,
    latest_event_delta: timedelta | None = None,
) -> dd.DataFrame:  # type: ignore
    meta = _remove_without_history(
        ddf._meta, time_treshold=time_treshold, latest_event_delta=latest_event_delta
    )
    return ddf.map_partitions(
        _remove_without_history,
        time_treshold=time_treshold,
        meta=meta,
        latest_event_delta=latest_event_delta,
    )


## Train/test split


def split_data(
    ddf: dd.DataFrame,  # type: ignore
    *,
    split_ratio: float = 0.8,
) -> tuple[dd.DataFrame, dd.DataFrame]:  # type: ignore
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
    array_to_pad: np.ndarray,  # type: ignore
    *,
    size: int,
) -> np.ndarray:  # type: ignore
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
    try:
        random_second = randrange(int_delta)  # nosec
    except ValueError:
        return end
    return start + timedelta(seconds=random_second)


def convert_datetime(time: str | pd.Timestamp) -> datetime:
    if isinstance(time, pd.Timestamp):
        return time.to_pydatetime()  # type: ignore
    try:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")


def _sample_user_histories_by_time(
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
            user_histories["OccurredTime"] <= t0
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

    logger.info("Partition processed.")

    return user_histories_sample


def sample_user_histories_by_time(
    ddf: dd.DataFrame,  # type: ignore
    *,
    min_time: datetime,
    max_time: datetime,
    history_size: int,
) -> pd.DataFrame:
    sample_for_meta = ddf.head(2, npartitions=-1)

    meta = _sample_user_histories_by_time(
        sample_for_meta,
        min_time=sample_for_meta["OccurredTime"].describe()["min"],
        max_time=sample_for_meta["OccurredTime"].describe()["max"],
        history_size=history_size,
    )
    sampled_data = ddf.map_partitions(
        _sample_user_histories_by_time,
        min_time=min_time,
        max_time=max_time,
        history_size=history_size,
        meta=meta,
    )
    return sampled_data


def _sample_user_histories_by_id(
    user_histories: pd.DataFrame,
    *,
    horizon_time: datetime,
    history_size: int,
) -> pd.DataFrame:
    filtered_index = user_histories[
        user_histories["OccurredTime"] < horizon_time
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


def sample_user_histories_by_id(
    ddf: dd.DataFrame,  # type: ignore
    *,
    horizon_time: datetime,
    history_size: int,
) -> dd.DataFrame:  # type: ignore
    meta = _sample_user_histories_by_id(
        ddf.head(2, npartitions=-1),
        horizon_time=horizon_time,
        history_size=history_size,
    )
    return ddf.map_partitions(
        _sample_user_histories_by_id,
        horizon_time=horizon_time,
        history_size=history_size,
        meta=meta,
    )


def split_train_val_test(
    ddf: dd.DataFrame,  # type: ignore
    *,
    latest_event_time: datetime,
    split_ratio: float = 0.8,
    churn_time: timedelta = timedelta(days=28),
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:  # type: ignore
    train_val = ddf[ddf["OccurredTime"] < latest_event_time - churn_time]
    train, validation = split_data(train_val, split_ratio=split_ratio)

    test = remove_without_history(ddf, time_treshold=latest_event_time - 2 * churn_time)

    return train, validation, test


def _preprocess_dataset(
    *,
    raw_data_path: Path,
    processed_data_path: Path,
    churn_time: timedelta = timedelta(days=28),
    history_size: int = 64,
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:  # type: ignore
    raw_data = dd.read_parquet(raw_data_path, calculate_divisions=True)  # type: ignore

    latest_event_time = convert_datetime(
        raw_data["OccurredTime"].describe().compute()["max"]
    )
    earliest_event_time = convert_datetime(
        raw_data["OccurredTime"].describe().compute()["min"]
    )

    train, validation, test = split_train_val_test(
        raw_data, latest_event_time=latest_event_time, churn_time=churn_time
    )

    train = sample_user_histories_by_time(
        train,
        history_size=history_size,
        min_time=earliest_event_time,
        max_time=latest_event_time - 2 * churn_time,
    )
    validation = sample_user_histories_by_time(
        validation,
        history_size=history_size,
        min_time=earliest_event_time,
        max_time=latest_event_time - 2 * churn_time,
    )
    test = sample_user_histories_by_id(
        test, horizon_time=latest_event_time - churn_time, history_size=history_size
    )

    train = write_and_read_parquet(train, path=processed_data_path / "train.parquet")
    validation = write_and_read_parquet(
        validation, path=processed_data_path / "validation.parquet"
    )
    test = write_and_read_parquet(test, path=processed_data_path / "test.parquet")

    return train, validation, test


def calculate_vocab(
    ddf: dd.DataFrame,  # type: ignore
    *,
    column: str,
    processed_data_path: Path,
    churn_time: timedelta = timedelta(days=28),
) -> None:
    latest_event_time = convert_datetime(
        ddf["OccurredTime"].describe().compute()["max"]
    )
    ddf = ddf[ddf["OccurredTime"] < latest_event_time - churn_time]
    vocabulary = list(ddf[column].unique().compute())
    with Path.open(processed_data_path / f"{column}_vocab.json", "w") as f:
        json.dump(vocabulary, f)


def calculate_time_mean_std(
    ddf: dd.DataFrame,  # type: ignore
    *,
    processed_data_path: Path,
    churn_time: timedelta = timedelta(days=28),
) -> None:
    latest_event_time = convert_datetime(
        ddf["OccurredTime"].describe().compute()["max"]
    )
    ddf = ddf[ddf["OccurredTime"] < latest_event_time - churn_time]
    time_mean, time_std = (
        ddf["OccurredTime"].compute().mean(),
        ddf["OccurredTime"].std().compute(),
    )

    time_stats = {
        "mean": int(time_mean.timestamp()),
        "std": int(time_std.total_seconds()),
    }

    with Path.open(processed_data_path / "time_stats.json", "w") as f:
        json.dump(time_stats, f)


def preprocess_dataset(raw_data_path: Path, processed_data_path: Path) -> None:
    logger.info("Starting preprocessing...")
    cluster = LocalCluster()  # type: ignore
    client = Client(cluster)  # type: ignore
    logger.info("Local cluster and client started.")

    logger.info(client)

    processed_data_path.mkdir(exist_ok=True, parents=True)
    logger.info("Created processed data path.")

    try:
        logger.info("Calculating vocab...")
        calculate_vocab(
            dd.read_parquet(raw_data_path),  # type: ignore
            column="DefinitionId",
            processed_data_path=processed_data_path,
        )
        logger.info("Calculating time mean and std...")
        calculate_time_mean_std(
            dd.read_parquet(raw_data_path),  # type: ignore
            processed_data_path=processed_data_path,
        )
        logger.info("Preprocessing dataset...")
        _preprocess_dataset(
            raw_data_path=raw_data_path,
            processed_data_path=processed_data_path,
        )
    finally:
        client.close()  # type: ignore
        cluster.close()  # type: ignore
    logger.info("Preprocessing done.")


if __name__ == "__main__":
    preprocess_dataset(raw_data_path, processed_data_path)
