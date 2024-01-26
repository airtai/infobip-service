import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy.engine import Connection

from infobip_service.download import (
    _create_clickhouse_connection_string,
    _pandas2dask,
    _pandas2dask_map,
    create_db_uri_for_clickhouse_datablob,
    fillna,
    get_clickhouse_connection,
    get_clickhouse_params_from_env_vars,
    get_unique_account_ids_model_ids,
)


def test_clickhouse_connection_string():
    actual = _create_clickhouse_connection_string(
        username="default",
        password="123456",
        host="localhost",
        port=8123,
        database="infobip",
        protocol="http",
    )
    assert actual == "clickhouse+http://default:123456@localhost:8123/infobip"

    actual = _create_clickhouse_connection_string(
        username="default",
        password="123456",
        host="localhost",
        port=9000,
        database="infobip",
        protocol="native",
    )
    assert actual == "clickhouse+native://default:123456@localhost:9000/infobip"

    actual = _create_clickhouse_connection_string(
        username="default",
        password="123?456@",
        host="localhost",
        port=9000,
        database="infobip",
        protocol="native",
    )
    assert (
        actual
        == "clickhouse+native://default:123%253F456%2540@localhost:9000/infobip"  # pragma: allowlist secret
    ), actual


def test_create_db_uri_for_clickhouse_datablob():
    actual = create_db_uri_for_clickhouse_datablob(
        username="default",
        password="123456",
        host="localhost",
        port=8123,
        database="infobip",
        table="events",
        protocol="http",
    )
    assert actual == "clickhouse+http://default:123456@localhost:8123/infobip/events"

    actual = create_db_uri_for_clickhouse_datablob(
        username="default",
        password="123456",
        host="localhost",
        port=9000,
        database="infobip",
        table="events",
        protocol="native",
    )
    assert actual == "clickhouse+native://default:123456@localhost:9000/infobip/events"

    actual = create_db_uri_for_clickhouse_datablob(
        username="default",
        password="123?456@",
        host="localhost",
        port=9000,
        database="infobip",
        table="events",
        protocol="native",
    )
    assert (
        actual
        == "clickhouse+native://default:123%253F456%2540@localhost:9000/infobip/events"  # pragma: allowlist secret
    ), actual


def test_get_clickhouse_params_from_env_vars(monkeypatch):
    # Set the required environment variables
    monkeypatch.setenv("KAFKA_CH_DATABASE", "infobip")
    monkeypatch.setenv("KAFKA_CH_HOST", "localhost")
    monkeypatch.setenv("KAFKA_CH_PASSWORD", "password")  # pragma: allowlist secret
    monkeypatch.setenv("KAFKA_CH_PORT", "9000")
    monkeypatch.setenv("KAFKA_CH_PROTOCOL", "native")
    monkeypatch.setenv("KAFKA_CH_TABLE", "events")
    monkeypatch.setenv("KAFKA_CH_USERNAME", "default")

    # Test case 1: Check if all required keys are present in the returned dictionary
    expected_keys = [
        "database",
        "host",
        "password",
        "port",
        "protocol",
        "table",
        "username",
    ]
    assert set(get_clickhouse_params_from_env_vars().keys()) == set(expected_keys)

    # Test case 2: Check if the values of the returned dictionary are of the correct types
    params = get_clickhouse_params_from_env_vars()
    assert isinstance(params["database"], str)
    assert isinstance(params["host"], str)
    assert isinstance(params["password"], str)
    assert isinstance(params["port"], int)
    assert isinstance(params["protocol"], str)
    assert isinstance(params["table"], str)
    assert isinstance(params["username"], str)


def test_fill_na():
    assert fillna("") == "''"
    assert fillna("Davor") == "'Davor'"
    assert fillna(None) == "''"


def create_duplicated_test_ddf():
    df = pd.DataFrame(
        {
            "AccountId": 12345,
            "PersonId": [1, 2, 2, 3, 3, 3],
            "OccurredTime": [
                datetime.fromisoformat(
                    f"2023-07-10T13:27:{i:02d}.{123456*(i+1) % 1_000_000:06d}"
                )
                for i in range(6)
            ],
            "DefinitionId": ["one"] * 3 + ["two"] * 2 + ["three"],
            "ApplicationId": None,
        }
    )
    df["OccurredTimeTicks"] = df["OccurredTime"].astype(int) // 1000
    df = pd.concat([df] * 3)
    df = df.sort_values(list(df.columns))
    return dd.from_pandas(df, npartitions=2)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Pyarrow types changed in Python 3.9 and higher"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows has different Pyarrow int casting for datetime",
)
def test_pandas2dask_map():
    ddf = create_duplicated_test_ddf()
    df = ddf.compute().set_index("PersonId")

    expected = pd.DataFrame(
        {
            "AccountId": [12345, 12345, 12345, 12345, 12345, 12345],
            "OccurredTime": [
                "2023-07-10 13:27:00.123456",
                "2023-07-10 13:27:01.246912",
                "2023-07-10 13:27:02.370368",
                "2023-07-10 13:27:03.493824",
                "2023-07-10 13:27:04.617280",
                "2023-07-10 13:27:05.740736",
            ],
            "DefinitionId": ["one", "one", "one", "two", "two", "three"],
            "ApplicationId": [None, None, None, None, None, None],
            "OccurredTimeTicks": [
                1688995620123456,
                1688995621246912,
                1688995622370368,
                1688995623493824,
                1688995624617280,
                1688995625740736,
            ],
        },
        index=pd.Index([1, 2, 2, 3, 3, 3], name="PersonId"),
    )
    expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
    expected["DefinitionId"] = expected["DefinitionId"].astype("string[pyarrow]")
    expected["ApplicationId"] = expected["ApplicationId"].astype("string[pyarrow]")

    actual = _pandas2dask_map(df)

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Pyarrow types changed in Python 3.9 and higher"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows has different Pyarrow int casting for datetime",
)
def test_pandas2dask():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ddf = create_duplicated_test_ddf()
        (d / "duplicated").mkdir()
        for i, partition in enumerate(ddf.partitions):
            partition.compute().to_parquet(d / "duplicated" / f"part_{i:06d}.parquet")

        _pandas2dask(d / "duplicated", d / "result")

        ddf = dd.read_parquet(d / "result")

        # display(ddf)
        # display(ddf.compute())

        expected = pd.DataFrame(
            {
                "AccountId": [12345, 12345, 12345, 12345, 12345, 12345],
                "OccurredTime": [
                    "2023-07-10 13:27:00.123456",
                    "2023-07-10 13:27:01.246912",
                    "2023-07-10 13:27:02.370368",
                    "2023-07-10 13:27:03.493824",
                    "2023-07-10 13:27:04.617280",
                    "2023-07-10 13:27:05.740736",
                ],
                "DefinitionId": ["one", "one", "one", "two", "two", "three"],
                "ApplicationId": [None, None, None, None, None, None],
                "OccurredTimeTicks": [
                    1688995620123456,
                    1688995621246912,
                    1688995622370368,
                    1688995623493824,
                    1688995624617280,
                    1688995625740736,
                ],
            },
            index=pd.Index([1, 2, 2, 3, 3, 3], name="PersonId"),
        )
        expected["OccurredTime"] = pd.to_datetime(expected["OccurredTime"])
        expected["DefinitionId"] = expected["DefinitionId"].astype("string[python]")
        expected["ApplicationId"] = expected["ApplicationId"].astype("string[python]")

        pd.testing.assert_frame_equal(ddf.compute(), expected)


@pytest.mark.skip(reason="This test requires a running ClickHouse instance.")
def test_get_clickhouse_connection():
    db_params = get_clickhouse_params_from_env_vars()

    with get_clickhouse_connection(
        **db_params,
    ) as connection:
        assert type(connection) == Connection

        query = "SELECT database, name from system.tables"
        df = pd.read_sql(sql=query, con=connection)

        database = db_params["database"]
        xs = df.loc[(df.database == db_params["database"]) & (df.name == "events")]
        if xs.shape[0] > 0:
            query = f"RENAME TABLE {database}.events TO {database}.events_distributed"
            _ = pd.read_sql(sql=query, con=connection)


def test_get_unique_account_ids_model_ids(monkeypatch):
    monkeypatch.setenv("KAFKA_CH_DATABASE", "infobip")
    monkeypatch.setenv("KAFKA_CH_HOST", "localhost")
    monkeypatch.setenv("KAFKA_CH_PASSWORD", "password")  # pragma: allowlist secret
    monkeypatch.setenv("KAFKA_CH_PORT", "9000")
    monkeypatch.setenv("KAFKA_CH_PROTOCOL", "native")
    monkeypatch.setenv("KAFKA_CH_TABLE", "events")
    monkeypatch.setenv("KAFKA_CH_USERNAME", "default")

    @contextmanager
    def mock_get_clickhouse_connection(*args, **kwargs):
        yield "mock_connection"

    def mock_read_sql(*args, **kwargs):
        return pd.DataFrame(
            {
                "AccountId": [1, 2, 3],
                "ApplicationId": ["a", "b", "c"],
                "ModelId": [1, 2, 3],
            }
        )

    # from infobip_service.download import pd
    monkeypatch.setattr(
        "infobip_service.download.get_clickhouse_connection",
        mock_get_clickhouse_connection,
    )
    monkeypatch.setattr(pd, "read_sql", mock_read_sql)

    df = get_unique_account_ids_model_ids()

    assert isinstance(df, list)
    assert len(df) == 3
    assert isinstance(df[0], dict)
    assert list(df[0].keys()) == ["AccountId", "ApplicationId", "ModelId"]
