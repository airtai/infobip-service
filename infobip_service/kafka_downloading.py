from pathlib import Path
from typing import Optional
from faststream import FastStream
from faststream.kafka import KafkaBroker
import random
from faststream.security import SASLScram256
import ssl

from os import environ

from datetime import datetime

from infobip_service.kafka_server import TrainingModelStatus, TrainingModelStart
from infobip_service.download import download_account_id_rows_as_parquet


group_id = environ.get("DOWNLOADING_GROUP_ID", None)
if group_id is None:
    group_id = f"infobip-downloader-{random.randint(100_000_000, 999_999_999):0,d}".replace(  # nosec: B311:blacklist
        ",", "-"
    )
print(f"{group_id=}")

root_path = Path(environ.get("ROOT_PATH")) if environ.get("ROOT_PATH") else None
if root_path is None:
    root_path = Path(".") / group_id

kwargs = dict(
    request_timeout_ms=120_000,
    max_batch_size=120_000,
    connections_max_idle_ms=10_000,
    # auto_offset_reset="earliest",
)

with_security = environ.get("WITH_SECURITY", "false").lower() == "true"
if with_security:
    ssl_context = ssl.create_default_context()
    security = SASLScram256(
        ssl_context=ssl_context,
        username=environ["KAFKA_API_KEY"],
        password=environ["KAFKA_API_SECRET"], 
    )
    kwargs["security"] = security

broker = KafkaBroker(f"{environ['KAFKA_HOSTNAME']}:{environ['KAFKA_PORT']}", **kwargs)

username = environ.get("USERNAME", "infobip")

@broker.publisher(f"{username}_training_model_status")
async def to_training_model_status(
    training_model_status: TrainingModelStatus,
) -> TrainingModelStatus:
    print(f"to_training_model_status({training_model_status})")
    return training_model_status


@broker.subscriber(f"{username}_training_model_start", auto_offset_reset="earliest")
async def on_training_model_start(
    msg: TrainingModelStart
) -> None:
    try:
        print(f"on_training_model_start({msg}) started")

        AccountId = msg.AccountId
        ApplicationId = msg.ApplicationId
        ModelId = msg.ModelId
        task_type = msg.task_type

        dt = datetime.now().date().isoformat()
        path = root_path / f"AccountId-{AccountId}" / f"ApplicationId-{ApplicationId}" / f"ModelId-{ModelId}" / dt
        
        
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
            print(
                f"on_training_model_start({msg}): path '{path}' exists, moving on..."
            )
        else:
            # this mean we can download data from clickhouse

            path.mkdir(parents=True, exist_ok=True)

            print(f"on_training_model_start({msg}): downloading data to '{path}'...")
            # with using_cluster("cpu"):
            download_account_id_rows_as_parquet(
                account_id=AccountId,
                application_id=ApplicationId,
                output_path=path,
            )

            print(f"on_training_model_start({msg}): data downloaded to '{path}'...")

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
        print(f"on_training_model_start({msg}) finished.")



app = FastStream(broker=broker)
