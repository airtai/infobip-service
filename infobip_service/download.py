import logging
import tempfile
from contextlib import contextmanager
from os import environ
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus as urlquote

import dask.dataframe as dd
import pandas as pd
from sqlalchemy.engine import Connection, create_engine

raw_data_path = Path() / ".." / "data" / "raw"


def _create_clickhouse_connection_string(
    username: str,
    password: str,
    host: str,
    port: int,
    database: str,
    protocol: str,
) -> str:
    # Double quoting is needed to fix a problem with special character '?' in password
    quoted_password = urlquote(urlquote(password))
    conn_str = (
        f"clickhouse+{protocol}://{username}:{quoted_password}@{host}:{port}/{database}"
    )

    return conn_str


def create_db_uri_for_clickhouse_datablob(
    username: str,
    password: str,
    host: str,
    port: int,
    table: str,
    database: str,
    protocol: str,
) -> str:
    """Create uri for clickhouse datablob based on connection params.

    Args:
        username: Username of clickhouse database
        password: Password of clickhouse database
        host: Host of clickhouse database
        port: Port of clickhouse database
        table: Table of clickhouse database
        database: Database to use
        protocol: Protocol to connect to clickhouse (native/http)

    Returns:
        An uri for the clickhouse datablob
    """
    clickhouse_uri = _create_clickhouse_connection_string(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        protocol=protocol,
    )
    clickhouse_uri = f"{clickhouse_uri}/{table}"
    return clickhouse_uri


def get_clickhouse_params_from_env_vars() -> Dict[str, Union[str, int]]:
    return {
        "username": environ["KAFKA_CH_USERNAME"],
        "password": environ["KAFKA_CH_PASSWORD"],
        "host": environ["KAFKA_CH_HOST"],
        "database": environ["KAFKA_CH_DATABASE"],
        "port": int(environ["KAFKA_CH_PORT"]),
        "protocol": environ["KAFKA_CH_PROTOCOL"],
        "table": environ["KAFKA_CH_TABLE"],
    }


@contextmanager  # type: ignore
def get_clickhouse_connection(  # type: ignore
    *,
    username: str,
    password: str,
    host: str,
    port: int,
    database: str,
    table: str,
    protocol: str,
    #     verbose: bool = False,
) -> Connection:
    if protocol != "native":
        raise ValueError()
    conn_str = _create_clickhouse_connection_string(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        protocol=protocol,
    )
    db_engine = create_engine(conn_str)
    with db_engine.connect() as connection:
        logging.info(f"Connected to database using {db_engine}")
        yield connection


def fillna(s: Optional[Any]) -> str:
    quote = "'"
    return f"{quote + ('' if s is None else str(s)) + quote}"


def _pandas2dask_map(
    df: pd.DataFrame, *, history_size: Optional[int] = None
) -> pd.DataFrame:
    df = df.reset_index()
    df = df.sort_values(["PersonId", "OccurredTime", "OccurredTimeTicks"])
    df = df.drop_duplicates()
    df = df.set_index("PersonId")
    return df


def _pandas2dask(
    downloaded_path: Path, output_path: Path, *, history_size: Optional[int] = None
) -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        ddf = dd.read_parquet(  # type: ignore
            downloaded_path,
            blocksize=None,
        )
        ddf["AccountId"] = ddf["AccountId"].astype("int64")

        # set index
        ddf = ddf.set_index("PersonId")
        ddf.to_parquet(d, engine="pyarrow")

        # deduplicate and sort by PersonId and OccurredTime
        ddf = dd.read_parquet(d)  # type: ignore

        ddf = ddf.map_partitions(_pandas2dask_map)

        ddf.to_parquet(output_path)


def _download_account_id_rows_as_parquet(
    *,
    account_id: Union[int, str],
    application_id: Optional[str],
    history_size: Optional[int] = None,
    host: str,
    port: int,
    username: str,
    password: str,
    database: str,
    protocol: str,
    table: str,
    chunksize: Optional[int] = 1_000_000,
    output_path: Path,
) -> None:
    with get_clickhouse_connection(  # type: ignore
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        table=table,
        protocol=protocol,
    ) as connection, tempfile.TemporaryDirectory() as td:
        d = Path(td)
        i = 0

        query = f"SELECT DISTINCT * FROM {table} WHERE AccountId={account_id}"  # nosec B608
        if application_id is not None and application_id != "":
            query = query + f" AND ApplicationId='{application_id}'"
        query = query + " ORDER BY PersonId ASC, OccurredTimeTicks DESC"
        if history_size:
            query = query + f" LIMIT {history_size} BY PersonId"

        logging.info(f"_download_account_id_rows_as_parquet(): {query=}")

        (d / "downloaded").mkdir(parents=True, exist_ok=True)
        for df in pd.read_sql(sql=query, con=connection, chunksize=chunksize):
            fname = d / "downloaded" / f"clickhouse_data_{i:09d}.parquet"
            logging.info(
                f"_download_account_id_rows_as_parquet() Writing data retrieved from the database to temporary file: {fname}"
            )
            df.to_parquet(fname, engine="pyarrow")  # type: ignore
            i = i + 1

        logging.info(
            f"_download_account_id_rows_as_parquet() Rewriting temporary parquet files from {d / 'clickhouse_data_*.parquet'} to output directory {output_path}"
        )
        _pandas2dask(d / "downloaded", output_path)

        # test if everything is ok
        dd.read_parquet(output_path).head()  # type: ignore


def download_account_id_rows_as_parquet(
    *,
    account_id: Union[int, str],
    application_id: Optional[str],
    history_size: Optional[int] = None,
    chunksize: Optional[int] = 1_000_000,
    output_path: Path,
) -> None:
    db_params = get_clickhouse_params_from_env_vars()

    return _download_account_id_rows_as_parquet(
        account_id=account_id,
        application_id=application_id,
        history_size=history_size,
        chunksize=chunksize,
        output_path=output_path,
        **db_params,  # type: ignore
    )


if __name__ == "__main__":
    AccountId = 12344
    ModelId = 20062
    ApplicationId = None

    raw_data_path.mkdir(exist_ok=True, parents=True)

    download_account_id_rows_as_parquet(
        account_id=AccountId,
        application_id=ApplicationId,
        output_path=raw_data_path,
    )

    ddf = dd.read_parquet(raw_data_path)  # type: ignore
    logging.info(f"{ddf.shape[0].compute()=:,d}")


def _get_unique_account_ids_model_ids(
    host: str,
    port: int,
    username: str,
    password: str,
    database: str,
    protocol: str,
    table: str,
) -> List[Dict[str, int]]:
    with get_clickhouse_connection(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        table=table,
        protocol=protocol,
    ) as connection:
        query = f"select DISTINCT on (AccountId, ModelId, ApplicationId) AccountId, ModelId, ApplicationId from {table}"  # nosec B608:hardcoded_sql_expressions
        df = pd.read_sql(sql=query, con=connection)
    return df.to_dict("records")  # type: ignore


def get_unique_account_ids_model_ids() -> List[Dict[str, int]]:
    db_params = get_clickhouse_params_from_env_vars()
    # Replace infobip_data with infobip_start_training_data for table param
    db_params["table"] = "infobip_start_training_data"
    return _get_unique_account_ids_model_ids(**db_params)  # type: ignore
