import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from infobip_service.dataset.preprocessing import (
    calculate_time_mean_std,
    calculate_vocab,
    get_histories_mask,
    remove_without_history,
    sample_dataframe_by_user_id_ddf,
    split_data,
    write_and_read_parquet,
)
from infobip_service.logger import get_logger, supress_timestamps

supress_timestamps(False)
logger = get_logger(__name__)

bucket_limits = (
    [np.timedelta64(0, "D")]
    + [np.timedelta64(days, "D") for days in range(1, 29)]
    + [np.timedelta64("NaT")]
)


def calculate_occured_timedelta(
    df: pd.DataFrame,
    t_max: datetime,
    churn_time: np.datetime64 = np.timedelta64(28, "D"),
) -> pd.DataFrame:
    xs = df["OccurredTime"].values
    x_max = np.max(xs)

    df.loc[:, "OccurredTimeDelta"] = np.concatenate(
        [xs[1:] - xs[:-1], [x_max - xs[-1]]]
    )

    ix = np.concatenate([df.index.values[1:] == df.index.values[:-1], [False]])
    df.loc[~ix, "OccurredTimeDelta"] = np.where(
        (t_max - df.loc[~ix, "OccurredTime"]) < churn_time,
        np.timedelta64("NaT"),
        churn_time,
    )

    df.loc[~df["OccurredTimeDelta"].isna(), "OccurredTimeDelta"] = np.minimum(
        df.loc[~df["OccurredTimeDelta"].isna(), "OccurredTimeDelta"], churn_time
    )

    df.loc[
        df["OccurredTime"] > t_max - churn_time, "OccurredTimeDelta"
    ] = np.timedelta64("NaT")

    if (df["OccurredTimeDelta"] > churn_time).any():
        raise ValueError("OccurredTimeDelta is greater than churn_time")

    return df


def get_bucket_rows(
    df: pd.DataFrame,
    *,
    floor: np.timedelta64,
    ceil: np.timedelta64 = np.timedelta64("NaT"),
) -> np.ndarray:
    if np.isnat(ceil):
        return df[df["OccurredTimeDelta"] >= floor]
    return df[(df["OccurredTimeDelta"] >= floor) & (df["OccurredTimeDelta"] < ceil)]


def sample_dataframe_buckets_last_event(
    df: pd.DataFrame,
    *,
    bucket_limits: list[np.timedelta64],
    sample_coefficient: float = 1.0,
) -> pd.DataFrame:
    n_samples = int((len(df) // (len(bucket_limits) - 1)) * sample_coefficient)

    sampled_buckets = [
        get_bucket_rows(df, floor=bucket_limits[i], ceil=bucket_limits[i + 1]).sample(
            n=n_samples, replace=True
        )
        for i in range(len(bucket_limits) - 1)
    ]

    return pd.concat(sampled_buckets)


def sample_dataframe_buckets(
    df: pd.DataFrame, *, bucket_limits: list[np.timedelta64], history_size
) -> pd.DataFrame:
    sampled_last_events = sample_dataframe_buckets_last_event(
        df, bucket_limits=bucket_limits, sample_coefficient=1 / history_size
    )
    sampled_last_events_index = sampled_last_events.index.values

    return get_histories_mask(df, sampled_last_events_index, history_size=history_size)


def calculate_bucket_sizes(
    ddf: dd.DataFrame, *, bucket_limits=list[np.timedelta64]
) -> dd.DataFrame:
    bucket_sizes = [
        float(
            ddf.map_partitions(
                get_bucket_rows, floor=bucket_limits[i], ceil=bucket_limits[i + 1]
            )
            .index.count()
            .compute()
        )
        for i in range(len(bucket_limits) - 1)
    ]

    return bucket_sizes


def _preprocess_dataset(
    *,
    raw_data_path: Path,
    processed_data_path: Path,
    churn_time: timedelta = timedelta(days=28),
    history_size: int = 32,
    drop_data_before: datetime | None = None,
    bucket_limits: list[np.timedelta64] = bucket_limits,
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, datetime]:  # type: ignore
    raw_ddf = dd.read_parquet(raw_data_path, calculate_divisions=True)  # type: ignore

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # convert datetime
        raw_ddf["OccurredTime"] = dd.to_datetime(raw_ddf["OccurredTime"])
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

        raw_with_timedeltas = raw_with_datetime_ddf.map_partitions(
            calculate_occured_timedelta, t_max=t_max
        )

        raw_with_timedeltas = write_and_read_parquet(
            raw_with_timedeltas, path=tmpdir_path / "raw_with_timedeltas.parquet"
        )

        logger.info("Calculating bucket sizes...")

        bucket_sizes = calculate_bucket_sizes(
            raw_with_timedeltas, bucket_limits=bucket_limits
        )

        logger.info("Splitting train/val/test dataset...")
        ts = datetime.now()

        train_val = raw_with_datetime_ddf[
            raw_with_datetime_ddf["OccurredTime"] < (t_max - churn_time)
        ]
        raw_train_ddf, raw_validation_ddf = split_data(train_val, split_ratio=0.8)

        raw_test_ddf = remove_without_history(
            raw_with_datetime_ddf,
            from_time=t_max - 2 * churn_time,
            to_time=t_max - churn_time,
        )
        raw_test_ddf = raw_test_ddf[
            raw_test_ddf["OccurredTime"] > t_max - 2 * churn_time
        ]

        raw_train_ddf = write_and_read_parquet(
            raw_train_ddf, path=tmpdir_path / "raw_train.parquet"
        )
        raw_validation_ddf = write_and_read_parquet(
            raw_validation_ddf, path=tmpdir_path / "raw_validation.parquet"
        )
        raw_test_ddf = write_and_read_parquet(
            raw_test_ddf, path=tmpdir_path / "raw_test.parquet"
        )

        logger.info(f" - train/val/test dataset split in {datetime.now() - ts}")

        logger.info("Calculating occurred time delta...")

        train_with_timedeltas_ddf = raw_train_ddf.map_partitions(
            calculate_occured_timedelta, t_max=t_max
        )
        train_with_timedeltas_ddf = write_and_read_parquet(
            train_with_timedeltas_ddf,
            path=tmpdir_path / "train_with_timedeltas.parquet",
        )

        validation_with_timedeltas_ddf = raw_validation_ddf.map_partitions(
            calculate_occured_timedelta, t_max=t_max
        )
        validation_with_timedeltas_ddf = write_and_read_parquet(
            validation_with_timedeltas_ddf,
            path=tmpdir_path / "validation_with_timedeltas.parquet",
        )

        test_with_timedeltas_ddf = raw_test_ddf.map_partitions(
            calculate_occured_timedelta, t_max=t_max
        )
        test_with_timedeltas_ddf = write_and_read_parquet(
            test_with_timedeltas_ddf, path=tmpdir_path / "test_with_timedeltas.parquet"
        )

        logger.info("Censoring data...")

        censored_train_ddf = train_with_timedeltas_ddf[
            train_with_timedeltas_ddf["OccurredTime"] < t_max - 2 * churn_time
        ]
        censored_train_ddf = censored_train_ddf.repartition(partition_size="100MB")
        censored_train_ddf = write_and_read_parquet(
            censored_train_ddf, path=tmpdir_path / "censored_train.parquet"
        )

        censored_validation_ddf = validation_with_timedeltas_ddf[
            validation_with_timedeltas_ddf["OccurredTime"] < t_max - 2 * churn_time
        ]
        censored_validation_ddf = censored_validation_ddf.repartition(
            partition_size="100MB"
        )
        censored_validation_ddf = write_and_read_parquet(
            censored_validation_ddf, path=tmpdir_path / "censored_validation.parquet"
        )

        logger.info("Resetting index...")
        censored_train_ddf = censored_train_ddf.reset_index()
        censored_train_ddf = write_and_read_parquet(
            censored_train_ddf, path=tmpdir_path / "censored_train_reset_index.parquet"
        )
        censored_validation_ddf = censored_validation_ddf.reset_index()
        censored_validation_ddf = write_and_read_parquet(
            censored_validation_ddf,
            path=tmpdir_path / "censored_validation_reset_index.parquet",
        )
        test_with_timedeltas_ddf.reset_index()
        test_with_timedeltas_ddf = write_and_read_parquet(
            test_with_timedeltas_ddf,
            path=tmpdir_path / "test_with_timedeltas_reset_index.parquet",
        )

        logger.info("Sampling data...")
        sampled_train_ddf = censored_train_ddf.map_partitions(
            sample_dataframe_buckets,
            bucket_limits=bucket_limits,
            history_size=history_size,
        )
        sampled_train_ddf = write_and_read_parquet(
            sampled_train_ddf, path=tmpdir_path / "train.parquet"
        )

        sampled_validation_ddf = censored_validation_ddf.map_partitions(
            sample_dataframe_buckets,
            bucket_limits=bucket_limits,
            history_size=history_size,
        )
        sampled_validation_ddf = write_and_read_parquet(
            sampled_validation_ddf, path=tmpdir_path / "validation.parquet"
        )

        sampled_test_ddf = sample_dataframe_by_user_id_ddf(
            test_with_timedeltas_ddf,
            history_size=history_size,
            t_max=t_max,
            construct_histories=True,
        )

        sampled_test_ddf = write_and_read_parquet(
            sampled_test_ddf, path=tmpdir_path / "test.parquet"
        )

        logger.info(f"Copying files to {processed_data_path}...")
        retval = ()
        for ds_name in [
            "train.parquet",
            "validation.parquet",
            "test.parquet",
        ]:
            if (processed_data_path / ds_name).exists():
                if (processed_data_path / ds_name).is_dir():
                    shutil.rmtree(processed_data_path / ds_name)
                else:
                    (processed_data_path / ds_name).unlink()
            shutil.copytree(tmpdir_path / ds_name, processed_data_path / ds_name)
            ddf = dd.read_parquet(processed_data_path / ds_name)
            retval = (*retval, ddf)
        logger.info(f" - files copied to {processed_data_path}...")

    return *retval, bucket_sizes, t_max


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
        train_ddf, _, _, bucket_sizes, _ = _preprocess_dataset(
            raw_data_path=raw_data_path,
            processed_data_path=processed_data_path,
        )

        with open(processed_data_path / "bucket_sizes.json", "w") as f:
            json.dump(bucket_sizes, f)

        logger.info("Calculating vocab...")
        calculate_vocab(
            train_ddf[train_ddf["HasHistory"] == True],  # type: ignore
            column="DefinitionId",
            processed_data_path=processed_data_path,
        )
        logger.info("Calculating time mean and std...")
        calculate_time_mean_std(
            train_ddf[train_ddf["HasHistory"] == True],  # type: ignore
            processed_data_path=processed_data_path,
        )

    finally:
        client.close()  # type: ignore
        cluster.close()  # type: ignore
    logger.info("Preprocessing done.")
