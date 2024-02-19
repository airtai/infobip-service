from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from infobip_service.dataset.load_dataset import UserHistoryDataset, prepare_sample


def test_prepare_sample():
    test_df = pd.DataFrame(
        {
            "DefinitionId": ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"],
            "OccurredTime": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "HasHistory": [False, True, True, True, True, True, True, True, True, True],
        }
    )

    time_mean = 1.0
    definition_id_vocabulary = ["A", "B", "C", "D", "E"]

    prepared_sample = prepare_sample(
        test_df, definition_id_vocabulary=definition_id_vocabulary, time_mean=time_mean
    )

    expected = np.array(
        [
            [5, 1],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [0, 5],
            [1, 6],
            [2, 7],
            [3, 8],
            [4, 9],
        ]
    )

    assert np.array_equal(prepared_sample, expected)


def test_user_history_dataset():
    test_df = pd.DataFrame(
        {
            "DefinitionId": ["A", "B", "C", "D", "E", "A"],
            "OccurredTime": [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                np.datetime64("2024-01-04"),
                np.datetime64("2024-01-05"),
                np.datetime64("2024-01-06"),
            ],
            "OccurredTimeDelta": [np.timedelta64(i, "D") for i in range(6)],
            "HasHistory": [False, True, True, False, False, True],
        }
    )

    time_mean = 1.0
    definition_id_vocabulary = ["A", "B", "C", "D", "E"]

    bins = [
        np.timedelta64(days, "D").astype("timedelta64[s]").astype(int)
        for days in range(0, 29, 1)
    ]

    with TemporaryDirectory() as temp_dir:
        test_df.to_parquet(Path(temp_dir) / "test.parquet")
        dataset = UserHistoryDataset(
            Path(temp_dir) / "test.parquet",
            definition_id_vocabulary=definition_id_vocabulary,
            time_mean=time_mean,
            bins=bins,
            history_size=3,
        )

    expected_x = np.array(
        [
            [5.0, 1.0],
            [
                1.0,
                np.datetime64("2024-01-02")
                .astype("datetime64[s]")
                .astype(int)
                .astype(float),
            ],
            [
                2.0,
                np.datetime64("2024-01-03")
                .astype("datetime64[s]")
                .astype(int)
                .astype(float),
            ],
        ]
    )

    expected_y = 2

    assert np.array_equal(dataset[0][0], expected_x)
    assert dataset[0][1] == expected_y
