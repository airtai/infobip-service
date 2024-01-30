import random
import ssl
from datetime import datetime
from os import environ
from pathlib import Path

from faststream import FastStream
from faststream.kafka import KafkaBroker
from faststream.security import SASLScram256

from infobip_service.dataset.download import download_account_id_rows_as_parquet
from infobip_service.kafka_server import TrainingModelStart, TrainingModelStatus
from infobip_service.logger import get_logger, supress_timestamps

supress_timestamps(False)
logger = get_logger(__name__)

downloading_group_id = environ.get("DOWNLOADING_GROUP_ID", None)
if downloading_group_id is None:
    downloading_group_id = (
        f"infobip-downloader-{random.randint(100_000_000, 999_999_999):0,d}".replace(  # nosec: B311:blacklist
            ",", "-"
        )
    )
logger.info(f"{downloading_group_id=}")

root_path = Path(environ.get("ROOT_PATH")) if environ.get("ROOT_PATH") else None  # type: ignore [arg-type]
if root_path is None:
    root_path = Path()
root_path = root_path / downloading_group_id

kwargs = {
    "request_timeout_ms": 120_000,
    "max_batch_size": 120_000,
    "connections_max_idle_ms": 10_000,
    # "group_id": downloading_group_id,
    # "auto_offset_reset": "earliest",
}

with_security = environ.get("WITH_SECURITY", "false").lower() == "true"
if with_security:
    ssl_context = ssl.create_default_context()
    security = SASLScram256(
        ssl_context=ssl_context,
        username=environ["KAFKA_API_KEY"],
        password=environ["KAFKA_API_SECRET"],
    )
    kwargs["security"] = security  # type: ignore [assignment]

bootstrap_servers = [
    f"{x}:{environ['KAFKA_PORT']}" for x in environ["KAFKA_HOSTNAME"].split(",")
]
broker = KafkaBroker(bootstrap_servers=bootstrap_servers, **kwargs)  # type: ignore [arg-type]

username = environ.get("USERNAME", "infobip")


# @broker.publisher(f"{username}_training_model_status")
async def to_training_model_status(
    training_model_status: TrainingModelStatus,
) -> None:
    logger.info(f"to_training_model_status({training_model_status})")
    await broker.publish(
        training_model_status, topic=f"{username}_training_model_status"
    )


@broker.subscriber(
    f"{username}_training_model_start",
    auto_offset_reset="earliest",
    group_id=downloading_group_id,
)
async def on_training_model_start(msg: TrainingModelStart) -> None:
    try:
        logger.info(f"on_training_model_start({msg}) started")

        AccountId = msg.AccountId  # noqa: N806
        ApplicationId = msg.ApplicationId  # noqa: N806
        ModelId = msg.ModelId  # noqa: N806

        dt = datetime.now().date().isoformat()
        path = (
            root_path  # type: ignore [operator]
            / f"AccountId-{AccountId}"
            / f"ApplicationId-{ApplicationId}"
            / f"ModelId-{ModelId}"
            / dt
        )

        training_model_status = TrainingModelStatus(
            AccountId=AccountId,
            ApplicationId=ApplicationId,
            ModelId=ModelId,
            current_step=0,
            current_step_percentage=0.0,
            total_no_of_steps=3,
        )
        await to_training_model_status(training_model_status)

        if path.exists():
            logger.info(
                f"on_training_model_start({msg}): path '{path}' exists, moving on..."
            )
        else:
            # this mean we can download data from clickhouse

            path.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"on_training_model_start({msg}): downloading data to '{path}'..."
            )
            # with using_cluster("cpu"):
            download_account_id_rows_as_parquet(
                account_id=AccountId,
                application_id=ApplicationId,
                output_path=path,
            )

            logger.info(
                f"on_training_model_start({msg}): data downloaded to '{path}'..."
            )

        training_model_status = TrainingModelStatus(
            AccountId=AccountId,
            ApplicationId=ApplicationId,
            ModelId=ModelId,
            current_step=0,
            current_step_percentage=1.0,
            total_no_of_steps=3,
        )
        await to_training_model_status(training_model_status)

    finally:
        logger.info(f"on_training_model_start({msg}) finished.")


app = FastStream(broker=broker)
