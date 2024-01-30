import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from faststream import FastStream
from faststream.kafka import TestKafkaBroker

from infobip_service.kafka_server import TrainingModelStart


def mock_download_account_id_rows_as_parquet(*args, **kwargs):
    return None


@pytest.mark.asyncio()
async def test_add_download_training_data(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTNAME", "localhost")
    monkeypatch.setenv("KAFKA_PORT", "9092")

    temp_username = "temp_username"
    monkeypatch.setenv("USERNAME", temp_username)
    from infobip_service.kafka_downloading import (
        broker,
        on_training_model_start,
    )

    with TemporaryDirectory(prefix="infobip_downloader_") as d:
        root_path = Path(d)
        monkeypatch.setenv("ROOT_PATH", str(root_path))
        import infobip_service.kafka_downloading

        monkeypatch.setattr(
            infobip_service.kafka_downloading,
            "download_account_id_rows_as_parquet",
            mock_download_account_id_rows_as_parquet,
        )

        _ = FastStream(broker=broker)

        AccountId = 317238  # noqa: N806
        ModelId = "10051"  # noqa: N806
        ApplicationId = "MobileApp"  # noqa: N806

        training_model_start = TrainingModelStart(
            AccountId=AccountId,
            ApplicationId=ApplicationId,
            ModelId=ModelId,
            no_of_records=1_000,
            task_type="churn",
        )

        async with TestKafkaBroker(broker, with_real=False) as br:
            await br.publish(
                training_model_start, topic=f"{temp_username}_training_model_start"
            )

            await asyncio.sleep(5)

            on_training_model_start.mock.assert_any_call(training_model_start.dict())
