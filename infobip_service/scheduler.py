import asyncio
from os import environ
from pathlib import Path

from aiokafka import AIOKafkaProducer
from redbird.logging import RepoHandler
from redbird.repos import CSVFileRepo
from rocketry import Rocketry
from rocketry.conds import daily, weekly
from rocketry.log import TaskRunRecord

from infobip_service.dataset.download import get_unique_account_ids_model_ids
from infobip_service.helpers import (
    get_aio_kafka_config,
)
from infobip_service.kafka_server import ModelTrainingRequest, StartPrediction

from .logger import get_logger

root_path = Path(environ["ROOT_PATH"])
log_file = root_path / "rocketry_logs.csv"

csv_file_repo = CSVFileRepo(filename=log_file, model=TaskRunRecord)
app = Rocketry(execution="async", logger_repo=csv_file_repo)

logger = get_logger(__name__)
handler = RepoHandler(repo=csv_file_repo)


@app.task(weekly)  # type: ignore
async def start_weekly_training() -> None:
    rows = get_unique_account_ids_model_ids()
    aio_kafka_config = get_aio_kafka_config()
    producer = AIOKafkaProducer(**aio_kafka_config)
    await producer.start()
    try:
        for row in rows:
            model_training_req = ModelTrainingRequest(
                AccountId=row["AccountId"],
                ApplicationId=row["ApplicationId"],  # type: ignore
                ModelId=row["ModelId"],  # type: ignore
                task_type="churn",  # type: ignore
                total_no_of_records=0,
            )
            msg = (model_training_req.json()).encode("utf-8")
            logger.info(f"Sending weekly retraining for {msg=}")
            await producer.send_and_wait("infobip_start_training_data", msg)
    finally:
        await producer.stop()


@app.task(daily)  # type: ignore
async def start_daily_prediction() -> None:
    rows = get_unique_account_ids_model_ids()
    aio_kafka_config = get_aio_kafka_config()
    producer = AIOKafkaProducer(**aio_kafka_config)
    await producer.start()
    try:
        for row in rows:
            start_prediction = StartPrediction(
                AccountId=row["AccountId"],
                ApplicationId=row["ApplicationId"],  # type: ignore
                ModelId=row["ModelId"],  # type: ignore
                task_type="churn",  # type: ignore
            )
            msg = (start_prediction.json()).encode("utf-8")
            logger.info(f"Sending daily prediction for {msg=}")
            await producer.send_and_wait("infobip_start_prediction", msg)
    finally:
        await producer.stop()


async def main() -> None:
    """Launch Rocketry app."""
    rocketry_task = asyncio.create_task(app.serve())
    await rocketry_task


if __name__ == "__main__":
    asyncio.run(main())
