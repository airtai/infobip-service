import dask.dataframe as dd
import numpy as np
import pandas as pd

from infobip_service.dataset.preprocessing_buckets import (
    calculate_bucket_sizes,
    get_bucket_rows,
    sample_dataframe_buckets,
    sample_dataframe_buckets_last_event,
)


def test_get_bucket_rows():
    test_df = pd.DataFrame(
        {"OccurredTimeDelta": pd.to_timedelta([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], unit="D")}
    )

    assert np.array_equal(
        get_bucket_rows(
            test_df, floor=np.timedelta64(0, "D"), ceil=np.timedelta64(3, "D")
        ).index.values,
        np.array([0, 1, 2]),
    )
    assert np.array_equal(
        get_bucket_rows(
            test_df, floor=np.timedelta64(3, "D"), ceil=np.timedelta64(6, "D")
        ).index.values,
        np.array([3, 4, 5]),
    )
    assert np.array_equal(
        get_bucket_rows(
            test_df, floor=np.timedelta64(6, "D"), ceil=np.timedelta64(9, "D")
        ).index.values,
        np.array([6, 7, 8]),
    )
    assert np.array_equal(
        get_bucket_rows(
            test_df, floor=np.timedelta64(8, "D"), ceil=np.timedelta64("NaT")
        ).index.values,
        np.array([8, 9]),
    )


def test_sample_dataframe_buckets_last_event():
    test_df = pd.DataFrame(
        {
            "OccurredTimeDelta": pd.to_timedelta(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], unit="D"
            ),
        }
    )

    test_ddf = dd.from_pandas(test_df, npartitions=1)

    bucket_limits = [
        np.timedelta64(0, "D"),
        np.timedelta64(5, "D"),
        np.timedelta64("NaT"),
    ]

    sampled_ddf = test_ddf.map_partitions(
        sample_dataframe_buckets_last_event, bucket_limits=bucket_limits, meta=test_ddf
    )

    assert (
        sampled_ddf.shape[0].compute() == 10
    ), f"{sampled_ddf.shape[0].compute():,d} == 10"

    assert calculate_bucket_sizes(sampled_ddf, bucket_limits=bucket_limits) == [5, 5]


def test_sample_dataframe_buckets():
    test_df = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "AccountId": [
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "FloatColumn": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "IntColumn": [1, 2, 3, 4, 5, 6],
            "OccurredTimeDelta": pd.to_timedelta([0, 1, 5, 2, 4, 6], unit="D"),
        }
    )

    bucket_limits = [
        np.timedelta64(0, "D"),
        np.timedelta64(3, "D"),
        np.timedelta64("NaT"),
    ]

    sampled_df = sample_dataframe_buckets(
        test_df, bucket_limits=bucket_limits, history_size=1
    )


def test_calculate_bucket_sizes():
    test_df = pd.DataFrame(
        {"OccurredTimeDelta": pd.to_timedelta([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], unit="D")}
    )

    test_ddf = dd.from_pandas(test_df, npartitions=2)

    assert calculate_bucket_sizes(
        test_ddf,
        bucket_limits=[
            np.timedelta64(0, "D"),
            np.timedelta64(3, "D"),
            np.timedelta64(6, "D"),
            np.timedelta64(9, "D"),
            np.timedelta64("NaT"),
        ],
    ) == [3, 3, 3, 1]
