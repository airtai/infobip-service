from infobip_service.kafka_server import TrainingModelStart

from faststream import FastStream
from faststream.kafka import KafkaBroker, TestKafkaBroker
import pytest

from tempfile import TemporaryDirectory
from pathlib import Path


def mock_download_account_id_rows_as_parquet(*args, **kwargs):
    return None

@pytest.mark.asyncio
async def test_add_download_training_data(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTNAME", "localhost")
    monkeypatch.setenv("KAFKA_PORT", "9092")

    temp_username = "temp_username"
    monkeypatch.setenv("USERNAME", temp_username)
    from infobip_service.kafka_downloading import broker , to_training_model_status, on_training_model_start
    with TemporaryDirectory(prefix="infobip_downloader_") as d:
        root_path = Path(d)
        monkeypatch.setenv("ROOT_PATH", str(root_path))
        import infobip_service.kafka_downloading
        monkeypatch.setattr(infobip_service.kafka_downloading, "download_account_id_rows_as_parquet", mock_download_account_id_rows_as_parquet)

        app = FastStream(broker=broker)

        AccountId = 317238
        ModelId = "10051"
        ApplicationId = "MobileApp"

        training_model_start = TrainingModelStart(
            AccountId=AccountId,
            ApplicationId=ApplicationId,
            ModelId=ModelId,
            no_of_records=1_000,
            task_type="churn",
        )

        async with TestKafkaBroker(broker, with_real=True) as br:
            await br.publish(training_model_start, topic=f"{temp_username}_training_model_start")

            on_training_model_start.mock.assert_any_call(training_model_start.model_dump())
