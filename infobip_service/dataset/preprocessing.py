import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from numpy import dtype, signedinteger

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
    ts = datetime.now()
    logger.info(f"Writing to {path}...")
    ddf.to_parquet(path, **kwargs)
    logger.info(f" - completed writing to {path} in {datetime.now() - ts}")
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
    from_time: datetime,
    to_time: datetime,
) -> pd.DataFrame:
    indexes_with_history = df[
        (df["OccurredTime"] < to_time) & (df["OccurredTime"] >= from_time)
    ].index

    return df[df.index.isin(indexes_with_history)]


def remove_without_history(
    ddf: dd.DataFrame,  # type: ignore
    *,
    from_time: datetime,
    to_time: datetime,
) -> dd.DataFrame:  # type: ignore
    meta = _remove_without_history(ddf._meta, from_time=from_time, to_time=to_time)
    return ddf.map_partitions(
        _remove_without_history,
        meta=meta,
        from_time=from_time,
        to_time=to_time,
    )


# Sample construction


def calculate_occured_timedelta(
    df: pd.DataFrame,
    t_max: datetime,
    churn_time: np.timedelta64 = np.timedelta64(28, "D"),  # noqa
) -> pd.DataFrame:
    xs = df["OccurredTime"].values
    x_max = np.max(xs)

    df["OccurredTimeDelta"] = np.concatenate([xs[1:] - xs[:-1], [x_max - xs[-1]]])

    ix = np.concatenate(
        [df["PersonId"].values[1:] == df["PersonId"].values[:-1], [False]]
    )
    df.loc[~ix, "OccurredTimeDelta"] = np.where(
        (t_max - df.loc[~ix, "OccurredTime"]) < churn_time,
        np.timedelta64("NaT"),
        churn_time,
    )

    df.loc[~df["OccurredTimeDelta"].isna(), "OccurredTimeDelta"] = np.minimum(
        df.loc[~df["OccurredTimeDelta"].isna(), "OccurredTimeDelta"], churn_time
    )

    df.loc[
        df["OccurredTime"] > t_max - churn_time, "OccurredTimeDelta"  # type: ignore
    ] = np.timedelta64("NaT")

    if (df["OccurredTimeDelta"] > churn_time).any():
        raise ValueError("OccurredTimeDelta is greater than churn_time")

    return df


def calculate_choice_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    p = (
        np.where(
            df["OccurredTimeDelta"].isna(),
            0.0,
            df["OccurredTimeDelta"].astype(int).astype(float),
        )
        * 1e-5
    )
    p = p / np.sum(p)
    df["Probability"] = p
    return df


def get_chosen_indexes(
    df: pd.DataFrame, *, num_choices: int
) -> np.ndarray[Any, dtype[signedinteger[Any]]]:
    return np.random.choice(
        len(df["Probability"]),
        size=num_choices,
        replace=True,
        p=df["Probability"].values,
    )


def get_histories_mask(
    df: pd.DataFrame, ix: np.ndarray[Any, dtype[signedinteger[Any]]], history_size: int
) -> pd.DataFrame:
    ih = np.arange(history_size) - history_size + 1
    ixx = (ix.reshape(-1, 1) + ih.reshape(1, -1)).reshape(-1)
    mask = np.zeros_like(ixx, dtype=bool)

    df.loc[:, "HasHistory"] = True

    user_ids = df.iloc[ixx]["PersonId"].values

    i = np.arange(history_size - 1, len(ixx), history_size)
    for j in range(1, history_size):
        mask[i - j] = (user_ids[i - j + 1] != user_ids[i - j]) | mask[i - j + 1]

    df = df.iloc[ixx]
    for c in df.columns:
        if c in ["PersonId", "AccountId", "HasHistory"]:
            continue
        df.loc[mask, c] = -1 if pd.api.types.is_integer_dtype(df[c]) else np.nan

    users_id = df["PersonId"].values
    df.loc[:, "PersonId"] = np.broadcast_to(
        users_id.reshape((-1, history_size))[:, -1],
        users_id.reshape(history_size, -1).shape,
    ).T.flatten()
    df.loc[:, "AccountId"] = np.broadcast_to(
        df["AccountId"].values.reshape((-1, history_size))[:, -1],
        df["AccountId"].values.reshape(history_size, -1).shape,
    ).T.flatten()
    df.loc[:, "HasHistory"] = ~mask.flatten()

    return df


def sample_dataframe_by_time(
    df: pd.DataFrame,
    *,
    history_size: int,
    t_max: datetime,
    churn_time: timedelta = timedelta(days=28),
    num_choices: int | None = None,
    construct_histories: bool,
) -> pd.DataFrame:
    df_len = len(df)
    df = df.reset_index()
    df = calculate_occured_timedelta(df, t_max - churn_time)
    df = calculate_choice_probabilities(df)
    if num_choices is None:
        num_choices = len(df) // history_size
    ix = get_chosen_indexes(df, num_choices=num_choices)
    if construct_histories:
        df = get_histories_mask(df, ix, history_size=history_size)

    if not np.isclose(len(df) / df_len, 1.0, atol=10 / num_choices):
        raise RuntimeError("Dataframe length changed")

    return df


def sample_dataframe_by_time_ddf(
    ddf: dd.DataFrame,  # type: ignore
    *,
    history_size: int,
    t_max: datetime,
    churn_time: timedelta = timedelta(days=28),
    construct_histories: bool,
) -> dd.DataFrame:  # type: ignore
    meta_sample = ddf.head(history_size * 10)
    meta = sample_dataframe_by_time(
        meta_sample,
        num_choices=1,
        history_size=history_size,
        t_max=meta_sample["OccurredTime"].max(),
        churn_time=churn_time,
        construct_histories=construct_histories,
    )

    return ddf.map_partitions(
        sample_dataframe_by_time,
        history_size=history_size,
        t_max=t_max,
        churn_time=churn_time,
        meta=meta,
        construct_histories=construct_histories,
    )


def get_last_valid_person_indexes(
    df: pd.DataFrame,
) -> np.ndarray[Any, dtype[signedinteger[Any]]]:
    valid_events = df["Probability"] != 0.0

    user_ids = df["PersonId"].values

    without_invalid = np.where(valid_events, user_ids, -1)
    breaks = np.concatenate([without_invalid[1:] != without_invalid[:-1], [True]])

    return np.where(breaks & valid_events)[0]


def sample_dataframe_by_user_id(
    df: pd.DataFrame,
    *,
    history_size: int,
    t_max: datetime,
    churn_time: timedelta = timedelta(days=28),
    construct_histories: bool,
) -> pd.DataFrame:
    df = df.reset_index()
    df = calculate_occured_timedelta(df, t_max)
    df = calculate_choice_probabilities(df)
    ix = get_last_valid_person_indexes(df)
    if construct_histories:
        df = get_histories_mask(df, ix, history_size=history_size)
    return df


def sample_dataframe_by_user_id_ddf(
    ddf: dd.DataFrame,  # type: ignore
    *,
    history_size: int,
    t_max: datetime,
    churn_time: timedelta = timedelta(days=28),
    construct_histories: bool,
) -> dd.DataFrame:  # type: ignore
    meta_sample = ddf.head(history_size * 10)
    meta = sample_dataframe_by_user_id(
        meta_sample,
        history_size=history_size,
        t_max=meta_sample["OccurredTime"].max(),
        churn_time=churn_time,
        construct_histories=construct_histories,
    )

    return ddf.map_partitions(
        sample_dataframe_by_user_id,
        history_size=history_size,
        t_max=t_max,
        churn_time=churn_time,
        meta=meta,
        construct_histories=construct_histories,
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


def _preprocess_dataset(
    *,
    raw_data_path: Path,
    processed_data_path: Path,
    churn_time: timedelta = timedelta(days=28),
    history_size: int = 32,
    drop_data_before: datetime | None = None,
    construct_histories: bool,
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, datetime]:  # type: ignore
    raw_ddf = dd.read_parquet(raw_data_path, calculate_divisions=True)  # type: ignore

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # convert datetime
        raw_ddf["OccurredTime"] = dd.to_datetime(raw_ddf["OccurredTime"])  # type: ignore
        raw_with_datetime_ddf = write_and_read_parquet(
            raw_ddf, path=tmpdir_path / "raw_with_datetime.parquet"
        )

        # trim old data
        if drop_data_before is not None:
            raw_with_datetime_ddf = raw_with_datetime_ddf[
                raw_with_datetime_ddf["OccurredTime"] > drop_data_before
            ]
            raw_with_datetime_ddf = write_and_read_parquet(
                raw_with_datetime_ddf,
                path=tmpdir_path / "raw_with_datetime_dropped.parquet",
            )

        logger.info("Calculating max time...")
        ts = datetime.now()
        t_max = raw_with_datetime_ddf["OccurredTime"].max().compute()
        logger.info(f" - max time calculated in {datetime.now() - ts}")

        logger.info("Splitting train/val/test dataset...")
        ts = datetime.now()

        train_val = raw_with_datetime_ddf[
            raw_with_datetime_ddf["OccurredTime"] < (t_max - churn_time)
        ]
        train, validation = split_data(train_val, split_ratio=0.8)

        test = remove_without_history(
            raw_with_datetime_ddf,
            from_time=t_max - 2 * churn_time,
            to_time=t_max - churn_time,
        )
        test = test[test["OccurredTime"] > t_max - 2 * churn_time]

        logger.info(f" - datasets split in {datetime.now() - ts}")

        train = write_and_read_parquet(train, path=tmpdir_path / "train-1.parquet")
        validation = write_and_read_parquet(
            validation, path=tmpdir_path / "validation-1.parquet"
        )
        test = write_and_read_parquet(test, path=tmpdir_path / "test-1.parquet")

        train = train.repartition(partition_size="100MB")
        validation = validation.repartition(partition_size="100MB")
        test = test.repartition(partition_size="100MB")

        train = write_and_read_parquet(train, path=tmpdir_path / "train-2.parquet")
        validation = write_and_read_parquet(
            validation, path=tmpdir_path / "validation-2.parquet"
        )
        test = write_and_read_parquet(test, path=tmpdir_path / "test-2.parquet")

        train = sample_dataframe_by_time_ddf(
            train,
            history_size=history_size,
            t_max=t_max,
            construct_histories=construct_histories,
        )
        validation = sample_dataframe_by_time_ddf(
            validation,
            history_size=history_size,
            t_max=t_max,
            construct_histories=construct_histories,
        )

        test = sample_dataframe_by_user_id_ddf(
            test,
            history_size=history_size,
            t_max=t_max,
            construct_histories=construct_histories,
        )

        train = write_and_read_parquet(train, path=tmpdir_path / "train.parquet")
        validation = write_and_read_parquet(
            validation, path=tmpdir_path / "validation.parquet"
        )
        test = write_and_read_parquet(test, path=tmpdir_path / "test.parquet")

        logger.info(f"Copying files to {processed_data_path}...")
        retval = ()
        for ds_name in ["train.parquet", "validation.parquet", "test.parquet"]:
            if (processed_data_path / ds_name).exists():
                if (processed_data_path / ds_name).is_dir():
                    shutil.rmtree(processed_data_path / ds_name)
                else:
                    (processed_data_path / ds_name).unlink()
            shutil.copytree(tmpdir_path / ds_name, processed_data_path / ds_name)
            ddf = dd.read_parquet(processed_data_path / ds_name)  # type: ignore
            retval = (*retval, ddf)  # type: ignore
        logger.info(f" - files copied to {processed_data_path}...")

    return *retval, t_max  # type: ignore


def calculate_vocab(
    ddf: dd.DataFrame,  # type: ignore
    *,
    column: str,
    processed_data_path: Path,
) -> None:
    vocabulary = list(ddf[column].unique().compute())
    with Path.open(processed_data_path / f"{column}_vocab.json", "w") as f:
        json.dump(vocabulary, f)


def calculate_time_mean_std(
    ddf: dd.DataFrame,  # type: ignore
    *,
    processed_data_path: Path,
) -> None:
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
        logger.info("Preprocessing dataset...")
        train_ddf, *_ = _preprocess_dataset(
            raw_data_path=raw_data_path,
            processed_data_path=processed_data_path,
            construct_histories=True,
        )

        logger.info("Calculating vocab...")
        calculate_vocab(
            train_ddf[train_ddf["HasHistory"]],  # type: ignore
            column="DefinitionId",
            processed_data_path=processed_data_path,
        )
        logger.info("Calculating time mean and std...")
        calculate_time_mean_std(
            train_ddf[train_ddf["HasHistory"]],  # type: ignore
            processed_data_path=processed_data_path,
        )

    finally:
        client.close()  # type: ignore
        cluster.close()  # type: ignore
    logger.info("Preprocessing done.")


if __name__ == "__main__":
    preprocess_dataset(raw_data_path, processed_data_path)
