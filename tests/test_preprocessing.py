from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd

from infobip_service.dataset.preprocessing import (
    _remove_without_history,
    calculate_choice_probabilities,
    calculate_occured_timedelta,
    get_chosen_indexes,
    get_histories_mask,
    get_last_valid_person_indexes,
    split_data,
)


def test_dataset_split():
    test_df = pd.DataFrame(
        {
            "user_id": np.random.randint(0, 100, 100),
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "label": np.random.randint(0, 2, 100),
        }
    )
    test_df = test_df.set_index("user_id")

    test_ddf = dd.from_pandas(test_df, npartitions=2)

    train_data, validation_data = split_data(test_ddf, split_ratio=0.8)

    train_len = train_data.shape[0].compute()
    validation_len = validation_data.shape[0].compute()
    test_len = test_df.shape[0]

    assert train_len + validation_len == test_len

    train_index = train_data.index.unique().compute()
    validation_index = validation_data.index.unique().compute()
    test_index = test_df.index.unique()

    assert len(np.intersect1d(train_index, validation_index)) == 0

    train_validation_index = np.concatenate([train_index, validation_index])

    assert len(np.intersect1d(train_validation_index, test_index)) == len(test_index)


def test_remove_without_history_1():
    test_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "OccurredTime": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2023, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 7),
                datetime(2023, 1, 1),
                datetime(2023, 12, 1),
                datetime(2023, 12, 4),
            ],
        }
    )

    test_df = test_df.set_index("user_id")

    from_time = datetime(2024, 1, 1)
    to_time = datetime(2024, 1, 4)

    result = _remove_without_history(test_df, from_time=from_time, to_time=to_time)

    assert result.shape[0] == 6

    assert result.index.unique().values.tolist() == [1, 2]


def test_remove_without_history_2():
    test_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "OccurredTime": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2023, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 7),
                datetime(2023, 1, 1),
                datetime(2023, 12, 1),
                datetime(2023, 12, 4),
            ],
        }
    )

    test_df = test_df.set_index("user_id")

    from_time = datetime(2023, 11, 30)
    to_time = datetime(2023, 12, 10)

    result = _remove_without_history(test_df, from_time=from_time, to_time=to_time)

    assert result.shape[0] == 3

    assert result.index.unique().values.tolist() == [3]


def test_calculating_timedeltas_1():
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
            "OccurredTime": [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
            ],
        }
    )

    test_df = calculate_occured_timedelta(test_df, np.datetime64("2024-02-03"))
    test_df.head(6)

    expected_timedeltas = np.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
            np.timedelta64(28, "D"),
            np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
            np.timedelta64(28, "D"),
        ]
    )

    np.testing.assert_array_equal(
        test_df["OccurredTimeDelta"].values, expected_timedeltas
    )


def test_calculating_timedeltas_2():
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
            "OccurredTime": [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
            ],
        }
    )

    test_df = calculate_occured_timedelta(
        test_df, np.datetime64("2024-01-03"), churn_time=np.timedelta64(1, "D")
    )

    expected_timedeltas = np.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
            np.timedelta64("NaT"),
            np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
            np.timedelta64("NaT"),
        ]
    )

    np.testing.assert_array_equal(
        test_df["OccurredTimeDelta"].values, expected_timedeltas
    )


def test_calculating_time_deltas_3():
    test_df = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "OccurredTime": [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-15"),
                np.datetime64("2024-01-30"),
                np.datetime64("2024-02-21"),
                np.datetime64("2024-02-27"),
            ],
        }
    )

    test_df = calculate_occured_timedelta(
        test_df, np.datetime64("2024-02-28"), churn_time=np.timedelta64(28, "D")
    )

    expected_timedeltas = np.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64(13, "D"),
            np.timedelta64(15, "D"),
            np.timedelta64(22, "D"),
            np.timedelta64("NaT"),
            np.timedelta64("NaT"),
        ]
    )

    np.testing.assert_array_equal(
        test_df["OccurredTimeDelta"].values, expected_timedeltas
    )


def test_calculating_timedeltas_4():
    test_df = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
            ],
            "OccurredTime": [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-15"),
                np.datetime64("2024-02-25"),
            ],
        }
    )

    test_df = calculate_occured_timedelta(
        test_df, np.datetime64("2024-04-30"), churn_time=np.timedelta64(28, "D")
    )

    expected_timedeltas = np.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64(13, "D"),
            np.timedelta64(28, "D"),
            np.timedelta64(28, "D"),
        ]
    )

    np.testing.assert_array_equal(
        test_df["OccurredTimeDelta"].values, expected_timedeltas
    )


def test_calculating_choice_probabilities_1():
    test_df = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "OccurredTimeDelta": [
                np.timedelta64(1, "D"),
                np.timedelta64(13, "D"),
                np.timedelta64(28, "D"),
                np.timedelta64(28, "D"),
                np.timedelta64("NaT"),
                np.timedelta64("NaT"),
            ],
        }
    )

    test_df = calculate_choice_probabilities(test_df)

    expected_probabilities = np.array([1 / 70, 13 / 70, 28 / 70, 28 / 70, 0, 0])

    np.testing.assert_array_almost_equal(
        test_df["Probability"].values, expected_probabilities
    )
    assert np.isclose(test_df["Probability"].sum(), 1)


def test_calculating_choice_probabilities_2():
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
            "OccurredTimeDelta": [
                np.timedelta64(1, "D"),
                np.timedelta64(1, "D"),
                np.timedelta64(28, "D"),
                np.timedelta64(1, "D"),
                np.timedelta64(1, "D"),
                np.timedelta64(28, "D"),
            ],
        }
    )

    test_df = calculate_choice_probabilities(test_df)

    expected_probabilities = np.array(
        [1 / 60, 1 / 60, 28 / 60, 1 / 60, 1 / 60, 28 / 60]
    )

    np.testing.assert_array_almost_equal(
        test_df["Probability"].values, expected_probabilities
    )
    assert np.isclose(test_df["Probability"].sum(), 1)


def test_get_chosen_indexes():
    test_dff = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "Probability": [1 / 70, 13 / 70, 28 / 70, 28 / 70, 0, 0],
        }
    )

    num_choices = 10000

    chosen_indexes = get_chosen_indexes(test_dff, num_choices=num_choices)

    assert chosen_indexes.min() >= 0
    assert chosen_indexes.max() < test_dff.shape[0]

    for index in range(test_dff.shape[0]):
        num_chosen_index = (chosen_indexes == index).sum()
        assert np.isclose(
            num_chosen_index / num_choices,
            test_dff["Probability"].values[index],
            atol=0.01,
        ), f"{num_chosen_index / num_choices} == {test_dff['Probability'].values[index]}"


def test_get_histories_mask():
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
            "OccuredTime": [
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                np.datetime64("2024-01-04"),
                np.datetime64("2024-01-05"),
                np.datetime64("2024-01-06"),
            ],
        }
    )

    history_size = 6
    ix = np.array([1, 5])

    test_df = get_histories_mask(test_df, ix, history_size)
    test_df.reset_index(drop=True, inplace=True)

    expected = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
            "AccountId": [
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
            "FloatColumn": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                2.0,
                np.nan,
                np.nan,
                np.nan,
                4.0,
                5.0,
                6.0,
            ],
            "IntColumn": [-1, -1, -1, -1, 1, 2, -1, -1, -1, 4, 5, 6],
            "OccuredTime": [
                np.datetime64("NaT"),
                np.datetime64("NaT"),
                np.datetime64("NaT"),
                np.datetime64("NaT"),
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                np.datetime64("NaT"),
                np.datetime64("NaT"),
                np.datetime64("NaT"),
                np.datetime64("2024-01-04"),
                np.datetime64("2024-01-05"),
                np.datetime64("2024-01-06"),
            ],
            "HasHistory": [
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
        }
    )

    pd.testing.assert_frame_equal(test_df, expected)


def test_get_last_valid_person_indexes_1():
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
            "Probability": [1 / 60, 1 / 60, 28 / 60, 1 / 60, 1 / 60, 28 / 60],
        }
    )

    last_valid_person_indexes = get_last_valid_person_indexes(test_df)

    assert last_valid_person_indexes.tolist() == [2, 5]


def test_get_last_valid_person_indexes_2():
    test_df = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "Probability": [1 / 70, 13 / 70, 28 / 70, 28 / 70, 0, 0],
        }
    )

    last_valid_person_indexes = get_last_valid_person_indexes(test_df)

    assert last_valid_person_indexes.tolist() == [3]


def test_get_last_valid_person_indexes_3():
    test_df = pd.DataFrame(
        {
            "PersonId": [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
            ],
            "Probability": [1 / 70, 13 / 70, 28 / 70, 0, 1 / 70, 13 / 70, 28 / 70, 0],
        }
    )

    last_valid_person_indexes = get_last_valid_person_indexes(test_df)

    assert last_valid_person_indexes.tolist() == [2, 6]
