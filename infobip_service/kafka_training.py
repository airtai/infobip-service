import asyncio
import random
import shutil
import ssl
import traceback
from datetime import datetime
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from faststream import FastStream
from faststream.kafka import KafkaBroker
from faststream.security import SASLScram256

from infobip_service.download import (
    download_account_id_rows_as_parquet,
    get_count_for_account_id,
)
from infobip_service.kafka_server import (
    ModelMetrics,
    ModelTrainingRequest,
    Prediction,
    StartPrediction,
    Tracker,
    TrainingDataStatus,
    TrainingModelStart,
    TrainingModelStatus,
)
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

training_group_id = environ.get("TRAINING_GROUP_ID", None)
if training_group_id is None:
    training_group_id = (
        f"infobip-trainer-{random.randint(100_000_000, 999_999_999):0,d}".replace(  # nosec: B311:blacklist
            ",", "-"
        )
    )
logger.info(f"{training_group_id=}")

root_path = Path(environ.get("ROOT_PATH")) if environ.get("ROOT_PATH") else None  # type: ignore [arg-type]
if root_path is None:
    root_path = Path()

downloading_root_path = root_path / downloading_group_id
training_root_path = root_path / training_group_id

kwargs = {
    "request_timeout_ms": 120_000,
    "max_batch_size": 120_000,
    "connections_max_idle_ms": 10_000,
    # "group_id": training_group_id,
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

# Variables

history_size = 32
batch_size = 256

downstream_epochs = 5
pretraining_epochs = 5


def get_paths(
    downloading_root_path: Path,
    training_root_path: Path,
    AccountId: int,
    ApplicationId: str | None,
    ModelId: str,
) -> dict[str, Path]:
    dt = datetime.now().date().isoformat()
    input_data_path = (
        downloading_root_path
        / f"AccountId-{AccountId}"
        / f"ApplicationId-{ApplicationId}"
        / f"ModelId-{ModelId}"
        / dt
    )

    pretraining_path = (
        training_root_path
        / f"AccountId-{AccountId}"
        / f"ApplicationId-{ApplicationId}"
        / f"ModelId-{ModelId}"
        / dt
        / "TimeSeriesMaskedPretrainingProblem"
    )

    training_path = (
        training_root_path
        / f"AccountId-{AccountId}"
        / f"ApplicationId-{ApplicationId}"
        / f"ModelId-{ModelId}"
        / dt
        / "TimeSeriesDownstreamDataChurn"
    )

    prediction_path = (
        training_root_path
        / f"AccountId-{AccountId}"
        / f"ApplicationId-{ApplicationId}"
        / f"ModelId-{ModelId}"
        / dt
        / "TimeSeriesDownstreamPrediction"
    )

    return dict(
        input_data_path=input_data_path,
        pretraining_path=pretraining_path,
        training_path=training_path,
        prediction_path=prediction_path,
    )


def custom_df_map_f(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["AccountId", "OccurredTimeTicks"])
    return df


# Training code

# def add_training(
#     broker: KafkaBroker,
#     *,
#     downloading_root_path: Path,
#     training_root_path: Path,
#     username: str = "infobip",
#     AccountId: Optional[int] = None,
#     ModelId: Optional[Union[int, str]] = None,
#     skip_all_requests: bool = False,
#     **kwargs: Any,
# ) -> None:
if not hasattr(broker, "premodels"):
    broker.premodels: dict[str, TimeSeriesMaskedPretrainingUNETSolution] = {}
if not hasattr(broker, "models"):
    broker.models: dict[str, TimeSeriesDownstreamSolution] = {}


@broker.publisher(f"{username}_training_model_status")
async def to_training_model_status(
    training_model_status: TrainingModelStatus,
) -> TrainingModelStatus:
    logger.info(f"to_training_model_status({training_model_status})")
    return training_model_status


@broker.publisher(f"{username}_model_metrics")
async def to_model_metrics(
    model_metrics: ModelMetrics,
) -> ModelMetrics:
    logger.info(f"to_model_metrics({model_metrics})")
    return model_metrics


@broker.publisher(f"{username}_start_prediction")
async def to_start_prediction(
    start_prediction: StartPrediction,
) -> StartPrediction:
    logger.info("*" * 100)
    logger.info(f"to_start_prediction({start_prediction})")
    logger.info("*" * 100)
    return start_prediction


async def pretrain(
    msg: TrainingModelStatus, broker: KafkaBroker = broker
) -> TimeSeriesMaskedPretrainingUNETSolution:
    # processing message

    AccountId = msg.AccountId
    ApplicationId = msg.ApplicationId
    ModelId = msg.ModelId

    paths = get_paths(
        downloading_root_path=downloading_root_path,
        training_root_path=training_root_path,
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
    )
    input_data_path = paths["input_data_path"]
    pretraining_path = paths["pretraining_path"]

    training_model_status = TrainingModelStatus(
        AccountId=AccountId,
        ModelId=ModelId,
        current_step=1,
        current_step_percentage=0.0,
        total_no_of_steps=3,
    )
    await to_training_model_status(training_model_status)

    if pretraining_path.exists() and (AccountId, ModelId) in broker.premodels:
        logger.info(
            f"on_training_model_status({msg})->pretrain(): path '{pretraining_path}' exists, moving on..."
        )

        pretrained_unet = broker.premodels[(AccountId, ModelId)]

    else:
        if pretraining_path.exists():
            logger.info(
                f"on_training_model_status({msg})->pretrain(): path '{pretraining_path}' exists, removing it and retraining it."
            )
            shutil.rmtree(pretraining_path, ignore_errors=True)
        pretraining_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"on_training_model_status({msg})->pretrain(): preparing data for pretraining in '{pretraining_path}'..."
        )

        try:
            pretrained_problem: TimeSeriesMaskedPretrainingProblem
            # with using_cpu_cluster():
            logger.info(
                dict(
                    input_data_path=input_data_path,
                    root_path=pretraining_path,
                    history_size=history_size,
                    custom_df_map_f=custom_df_map_f,
                    mask_column="padding_mask",
                    rate=0.25,
                    row_rate=0.25,
                    column_rate=0.25,
                )
            )
            pretrained_problem = TimeSeriesMaskedPretrainingProblem.from_parquet_files(
                input_data_path=input_data_path,
                root_path=pretraining_path,
                history_size=history_size,
                custom_df_map_f=custom_df_map_f,
                mask_column="padding_mask",
                rate=0.25,
                row_rate=0.25,
                column_rate=0.25,
            )
            logger.info(
                f"on_training_model_status({msg})->pretrain(): {pretrained_problem=}"
            )
        except Exception as e:
            logger.info(
                f"on_training_model_status({msg})->pretrain(): TimeSeriesMaskedPretrainingProblem.from_parquet_files() failed!"
            )
            traceback.print_exc()
            logger.error(
                f"on_training_model_status({msg})->pretrain(): TimeSeriesMaskedPretrainingProblem.from_parquet_files() failed!"
            )
            raise e

        training_model_status = TrainingModelStatus(
            AccountId=AccountId,
            ModelId=ModelId,
            current_step=1,
            current_step_percentage=0.1,
            total_no_of_steps=3,
        )
        await to_training_model_status(training_model_status)

        pretrained_unet = TimeSeriesMaskedPretrainingUNETSolution(
            problem=pretrained_problem,
            batch_size=batch_size,
            epochs=pretraining_epochs,
        )
        logger.info(
            f"on_training_model_status({msg})->pretrain(): pretrained problem created: {pretrained_unet=}"
        )

        logger.info(
            f"on_training_model_status({msg})->pretrain(): pretraining data for {pretraining_epochs} epochs..."
        )
        try:
            pretrained_unet.pretrain(
                verbose=2,  # type: ignore
                **kwargs,
            )
        except Exception as e:
            logger.error(
                f"on_training_model_status({msg})->pretrain(): pretraining failed: {e=}"
            )
            raise e

        broker.premodels[(AccountId, ModelId)] = pretrained_unet

    training_model_status = TrainingModelStatus(
        AccountId=AccountId,
        ModelId=ModelId,
        current_step=1,
        current_step_percentage=1.0,
        total_no_of_steps=3,
    )
    await to_training_model_status(training_model_status)

    return pretrained_unet


async def train(
    msg: TrainingModelStatus,
    pretrained_unet: TimeSeriesMaskedPretrainingUNETSolution,
    broker: KafkaBroker = broker,
) -> None:
    # processing message
    AccountId = msg.AccountId
    ApplicationId = msg.ApplicationId
    ModelId = msg.ModelId

    paths = get_paths(
        downloading_root_path=downloading_root_path,
        training_root_path=training_root_path,
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
    )
    input_data_path = paths["input_data_path"]
    training_path = paths["training_path"]

    training_model_status = TrainingModelStatus(
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
        current_step=2,
        current_step_percentage=0.0,
        total_no_of_steps=3,
    )
    await to_training_model_status(training_model_status)

    if training_path.exists() and (AccountId, ModelId) in broker.models:
        logger.info(
            f"on_training_model_status({msg})->train(): path '{training_path}' exists, moving on..."
        )
        churn_unet_solution = broker.models[(AccountId, ModelId)]
    else:
        if training_path.exists():
            logger.info(
                f"on_training_model_status({msg})->train(): path '{training_path}' exists, removing it and retraining it."
            )
            shutil.rmtree(training_path, ignore_errors=True)
        training_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"on_training_model_status({msg})->train(): preparing data for training in '{training_path}'..."
        )
        downstream_problem: TimeSeriesDownstreamProblem

        # with using_cpu_cluster():
        downstream_problem = TimeSeriesDownstreamProblem.from_parquet_files(
            input_data_path=input_data_path,
            root_path=training_path,
            history_size=history_size,
            custom_df_map_f=custom_df_map_f,
            application_id=ApplicationId,
        )
        logger.info(f"on_training_model_status({msg})->train(): {downstream_problem=}")

        training_model_status = TrainingModelStatus(
            AccountId=AccountId,
            ModelId=ModelId,
            current_step=2,
            current_step_percentage=0.1,
            total_no_of_steps=3,
        )
        await to_training_model_status(training_model_status)

        churn_unet_solution = TimeSeriesDownstreamSolution(
            root_path=training_path,
            problem=downstream_problem,
            pretrained_solution=pretrained_unet,
            epochs=downstream_epochs,
            batch_size=batch_size,
        )

        logger.info(
            f"on_training_model_status({msg})->train(): trained problem created: {churn_unet_solution=}"
        )

        logger.info(
            f"on_training_model_status({msg})->train(): training data for {churn_unet_solution} epochs..."
        )
        try:
            churn_unet_solution.train(
                verbose=2,  # type: ignore
                **kwargs,
            )

        except Exception as e:
            logger.error(
                f"on_training_model_status({msg})->train(): training failed: {e=}"
            )
            raise e

        broker.models[(AccountId, ModelId)] = churn_unet_solution

    training_model_status = TrainingModelStatus(
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
        current_step=2,
        current_step_percentage=1.0,
        total_no_of_steps=3,
    )
    await to_training_model_status(training_model_status)

    # send metrics
    accuracy = churn_unet_solution.val_accuracy
    model_metrics = ModelMetrics(
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
        task_type="churn",
        timestamp=datetime.now(),
        auc=accuracy,
        recall=0.0,
        precission=0.0,
        accuracy=accuracy,
        f1=0.0,
    )
    await to_model_metrics(model_metrics)
    logger.info(
        f"on_training_model_status({msg})->train(): metrics sent: {model_metrics=}"
    )

    start_prediction = StartPrediction(
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
        task_type="churn",
    )
    await to_start_prediction(start_prediction)
    logger.info(
        f"on_training_model_status({msg})->train(): start prediction sent: {start_prediction=}"
    )


@broker.subscriber(
    f"{username}_training_model_status",
    auto_offset_reset="earliest",
    group_id=training_group_id,
)
async def on_training_model_status(
    msg: TrainingModelStatus,
    broker: KafkaBroker = broker,
    AccountId: int | None = None,
    ModelId: int | str | None = None,
    skip_all_requests: bool = False,
) -> None:
    #         return
    try:
        logger.info(f"on_training_model_status({msg}) started")

        if skip_all_requests:
            logger.info(
                f"on_training_model_status({msg}) skipping due to skip_all_requests set to True..."
            )
            return

        if msg.current_step != 0 or msg.current_step_percentage != 1.0:
            logger.info(
                f"on_training_model_status({msg}) skipping due to {msg.current_step=} and {msg.current_step_percentage=}..."
            )
            return

        if AccountId is not None and msg.AccountId != AccountId:
            logger.info(
                f"on_training_model_status({msg}) skipping due to {msg.AccountId=}, {AccountId=}"
            )
            return

        if ModelId is not None and msg.ModelId != ModelId:
            logger.info(
                f"on_training_model_status({msg}) Skipping due to {msg.ModelId=}, {ModelId=}"
            )
            return

        pretrained_unet = await pretrain(msg)
        await train(msg, pretrained_unet=pretrained_unet)

        logger.info(f"on_training_model_status({msg}) SUCCESSFULLY finished.")
    except Exception as e:
        traceback.print_exc()
        raise e
    finally:
        logger.info(f"on_training_model_status({msg}) finished.")


# Prediction

prediction = None

# def add_predictions(
#     app: FastKafka,
#     *,
#     downloading_root_path: Path,
#     training_root_path: Path,
#     username: str = "infobip",
#     AccountId: Optional[int] = None,
#     ModelId: Optional[Union[int, str]] = None,
#     skip_all_requests: bool = False,
# ) -> None:


@broker.publisher(f"{username}_start_training_data")
async def to_start_training_data(
    request: ModelTrainingRequest,
) -> ModelTrainingRequest:
    logger.info(f"to_start_training_data({request})")
    return request


@broker.publisher(f"{username}_prediction")
async def to_prediction(
    prediction: Prediction,
) -> Prediction:
    #         logger.info(f"to_prediction({prediction})")
    return prediction


@broker.subscriber(
    f"{username}_start_prediction",
    auto_offset_reset="earliest",
    group_id=training_group_id,
)
async def on_start_prediction(
    msg: StartPrediction,
    broker: KafkaBroker = broker,
    skip_all_requests: bool = False,
) -> None:
    global prediction
    try:
        if skip_all_requests:
            logger.info(
                f"on_training_model_status({msg}) skipping due to skip_all_requests set to True..."
            )
            return

        AccountId = msg.AccountId
        ApplicationId = msg.ApplicationId
        ModelId = msg.ModelId
        task_type = msg.task_type
        prediction_time = datetime.now()

        if (AccountId, ModelId) not in broker.models:
            request = ModelTrainingRequest(
                AccountId=AccountId,
                ApplicationId=ApplicationId,
                ModelId=ModelId,
                task_type=task_type,
                total_no_of_records=0,
            )
            logger.info(
                f"on_start_prediction({msg}) no model found, making training request {request=}"
            )
            await to_start_training_data(request)
            return

        churn_unet_solution = broker.models[(AccountId, ModelId)]

        if AccountId is not None and msg.AccountId != AccountId:
            logger.info(
                f"on_start_prediction({msg}) skipping due to {msg.AccountId=}, {AccountId=}"
            )
            return

        if ModelId is not None and msg.ModelId != ModelId:
            logger.info(
                f"on_start_prediction({msg}) Skipping due to {msg.ModelId=}, {ModelId=}"
            )
            return

        paths = get_paths(
            downloading_root_path=downloading_root_path,
            training_root_path=training_root_path,
            AccountId=AccountId,
            ApplicationId=ApplicationId,
            ModelId=ModelId,
        )
        prediction_path = paths["prediction_path"]

        with TemporaryDirectory() as d:
            prediction_input_data_path = Path(d)

            logger.info(
                f"on_start_prediction({msg}): downloading prediction data to '{prediction_input_data_path}'..."
            )
            # with using_cluster("cpu"):
            download_account_id_rows_as_parquet(
                account_id=AccountId,
                output_path=prediction_input_data_path,
                history_size=history_size,
                application_id=ApplicationId,
            )

            logger.info(
                f"on_start_prediction({msg}): prediction data downloaded to '{prediction_input_data_path}'..."
            )

            # make predictions
            prediction_time = datetime.now()
            # with using_cpu_cluster():
            churn_prediction_data: TimeSeriesDownstreamPrediction = (
                TimeSeriesDownstreamPrediction.from_parquet_files(
                    input_data_path=prediction_input_data_path,
                    root_path=prediction_path,
                    history_size=history_size,
                    custom_df_map_f=custom_df_map_f,
                    timestamp_column="OccurredTime",
                )
            )

        # todo: remove debugging stuff
        broker.churn_prediction_data = churn_prediction_data
        from traceback_with_variables import printing_exc

        with printing_exc():
            prediction = churn_unet_solution.predict(
                prediction_data=churn_prediction_data
            )

        # check for NA values
        df = prediction[prediction["prediction"].isna()]
        if df.shape[0] > 0:
            logger.info(
                f"on_start_prediction({msg}): predictions have undefined values {df}"
            )

        logger.info(f"{prediction.head()=}")
        logger.info(f"{prediction.tail()=}")

        for i, x in (
            prediction[~prediction["prediction"].isna()].reset_index().iterrows()
        ):
            p = Prediction(
                AccountId=AccountId,
                ApplicationId=ApplicationId,
                ModelId=ModelId,
                PersonId=x["PersonId"],
                prediction_time=prediction_time,
                task_type="churn",
                score=x["prediction"],
            )
            await to_prediction(p)
        logger.info(
            f"on_start_prediction({msg}): predictions sent, the last one being: {p}"
        )
    except Exception as e:
        logger.info(f"on_start_prediction({msg}): predictions failed - {e!s}")
        logger.info(f"on_start_prediction({msg}): predictions failed, moving on...")


# Airt Service part

# def add_process_start_training_data(
#     app: FastKafka,
#     *,
#     username: str = "infobip",
#     stop_on_no_change_interval: int = 60,
#     abort_on_no_change_interval: int = 120,
#     sleep_interval: int = 5,
# ) -> None:

sleep_interval = 5
stop_on_no_change_interval = 60
abort_on_no_change_interval = 120


@broker.publisher(f"{username}_training_data_status")
async def to_training_data_status(
    training_data_status: TrainingDataStatus,
) -> TrainingDataStatus:
    print(f"to_training_data_status({training_data_status})")
    return training_data_status

@broker.publisher(f"{username}_training_model_start")
async def to_training_model_start(
    training_model_start: TrainingModelStart,
) -> TrainingModelStart:
    print(f"to_training_model_start({training_model_start})")
    return training_model_start

@broker.subscriber(
        f"{username}_start_training_data",
        auto_offset_reset="earliest",
        group_id=training_group_id,
)
async def on_start_training_data(
    msg: ModelTrainingRequest
) -> None:
    logger.info(msg, "on_start_training_data() starting...")

    account_id = msg.AccountId
    application_id = msg.ApplicationId
    model_id = msg.ModelId
    total_no_of_records = msg.total_no_of_records

    tracker = Tracker(
        limit=total_no_of_records,
        timeout=stop_on_no_change_interval,
        abort_after=abort_on_no_change_interval,
    )

    while not tracker.finished():
        curr_count, timestamp = get_count_for_account_id(
            account_id=account_id,
        )
        if curr_count is not None:
            if tracker.update(curr_count):
                training_data_status = TrainingDataStatus(
                    no_of_records=curr_count, **msg.model_dump()
                )
                await to_training_data_status(training_data_status)
        else:
            await logger.warning(
                msg,
                "on_start_training_data(): no data yet received in the database.",
            )

        await asyncio.sleep(sleep_interval)

    if tracker.aborted():
        await logger.error(msg, "on_start_training_data(): data retrieval aborted!")
    else:
        # trigger model training start
        training_model_start = TrainingModelStart(
            no_of_records=curr_count, **msg.model_dump()
        )
        await to_training_model_start(training_model_start)

        await logger.info(msg, "on_start_training_data(): finished")


app = FastStream(broker=broker)
