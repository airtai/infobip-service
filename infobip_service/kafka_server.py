from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field, NonNegativeInt

from infobip_service.logger import get_logger, supress_timestamps

get_count_for_account_id: Callable[[int, str], tuple[int, datetime]] | None = None
get_all_person_ids_for_account_id: Callable[[int, str], list[int]] | None = None


supress_timestamps(False)
logger = get_logger(__name__)


class LogMessage(BaseModel):
    """Info, error and warning messages."""

    level: NonNegativeInt = Field(
        10, json_schema_extra={"example": 10, "description": "level of the message"}
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, json_schema_extra={"description": "timestamp"}
    )
    message: str = Field(
        ...,
        json_schema_extra={"example": "something went wrong", "description": "message"},
    )

    class Config:
        json_encoders = {  # noqa: RUF012
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S"),
        }


class TaskType(str, Enum):
    """TaskType is an enumeration of supported model types."""

    churn = "churn"
    propensity_to_buy = "propensity_to_buy"


class ModelTrainingRequest(BaseModel):
    """Request to start training a model."""

    AccountId: NonNegativeInt = Field(
        ..., json_schema_extra={"example": 202020, "description": "ID of an account"}
    )
    ApplicationId: str | None = Field(
        default=None,
        json_schema_extra={
            "example": "TestApplicationId",
            "description": "Id of the application in case there is more than one for the AccountId",
        },
    )
    ModelId: str = Field(
        ...,
        json_schema_extra={
            "example": "ChurnModelForDrivers",
            "description": "User supplied ID of the model trained",
        },
    )
    task_type: TaskType = Field(
        ...,
        json_schema_extra={
            "description": "Model type, only 'churn' is supported right now"
        },
    )
    total_no_of_records: NonNegativeInt = Field(
        ...,
        json_schema_extra={
            "example": 1_000_000,
            "description": "approximate total number of records (rows) to be ingested",
        },
    )


class EventData(BaseModel):
    """A sequence of events for a fixed account_id."""

    AccountId: NonNegativeInt = Field(
        ..., json_schema_extra={"example": 202020, "description": "ID of an account"}
    )
    ApplicationId: str | None = Field(
        default=None,
        json_schema_extra={
            "example": "TestApplicationId",
            "description": "Id of the application in case there is more than one for the AccountId",
        },
    )
    ModelId: str = Field(
        default=...,
        json_schema_extra={
            "example": "ChurnModelForDrivers",
            "description": "User supplied ID of the model trained",
        },
    )

    DefinitionId: str = Field(
        ...,
        json_schema_extra={
            "example": "appLaunch",
            "description": "name of the event",
        },
        min_length=1,
    )
    OccurredTime: datetime = Field(
        ...,
        json_schema_extra={
            "example": "2021-03-28T00:34:08",
            "description": "local time of the event",
        },
    )
    OccurredTimeTicks: NonNegativeInt = Field(
        ...,
        json_schema_extra={
            "example": 1616891648496,
            "description": "local time of the event as the number of ticks",
        },
    )
    PersonId: NonNegativeInt = Field(
        ..., json_schema_extra={"example": 12345678, "description": "ID of a person"}
    )

    class Config:
        json_encoders = {  # noqa: RUF012
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S"),
        }


class RealtimeData(EventData):
    """A sequence of events for a fixed account_id."""

    pass


class TrainingDataStatus(BaseModel):
    """Status of the training data."""

    AccountId: NonNegativeInt = Field(
        ..., examples=[202020], description="ID of an account"
    )
    ModelId: str = Field(
        ...,
        examples=["ChurnModelForDrivers"],
        description="User supplied ID of the model trained",
    )

    no_of_records: NonNegativeInt = Field(
        ...,
        examples=[12_345],
        description="number of records (rows) ingested",
    )
    total_no_of_records: NonNegativeInt = Field(
        ...,
        examples=[1_000_000],
        description="total number of records (rows) to be ingested",
    )


class TrainingModelStart(BaseModel):
    """Request to start training a model."""

    AccountId: NonNegativeInt = Field(
        ..., examples=[202020], description="ID of an account"
    )
    ApplicationId: str | None = Field(
        default=None,
        examples=["TestApplicationId"],
        description="Id of the application in case there is more than one for the AccountId",
    )
    ModelId: str = Field(
        ...,
        examples=["ChurnModelForDrivers"],
        description="User supplied ID of the model trained",
    )
    task_type: TaskType = Field(
        ..., description="Model type, only 'churn' is supported right now"
    )
    no_of_records: NonNegativeInt = Field(
        ...,
        examples=[1_000_000],
        description="number of records (rows) in the DB used for training",
    )


def get_key(msg: BaseModel, attrs: list[str] | None = None) -> bytes:
    if attrs is None:
        attrs = ["AccountId", "ModelId"]

    sx = [
        f"{attr}='{getattr(msg, attr)}'" if hasattr(msg, attr) else "" for attr in attrs
    ]

    return ", ".join(sx).encode("utf-8")


class Tracker:
    """Tracks the progress of the training data ingestion."""

    def __init__(self, *, limit: int, timeout: int, abort_after: int):
        """Initializes the tracker."""
        self._limit = limit
        self._timeout = timeout
        self._abort_after = abort_after
        self._count: int | None = None
        self._last_updated: datetime | None = None
        self._sterted_at: datetime = datetime.now()

    def update(self, count: int) -> bool:
        if self._count != count:
            self._count = count
            self._last_updated = datetime.now()
            return True
        else:
            return False

    def finished(self) -> bool:
        if self._count is not None:
            return (self._count >= self._limit) or (
                datetime.now() - self._last_updated  # type: ignore
            ) > timedelta(seconds=self._timeout)
        else:
            return self.aborted()

    def aborted(self) -> bool:
        return self._count is None and (datetime.now() - self._sterted_at) > timedelta(
            seconds=self._abort_after
        )


class TrainingModelStatus(BaseModel):
    """Status of the training model."""

    AccountId: NonNegativeInt = Field(
        ..., examples=[202020], description="ID of an account"
    )
    ApplicationId: str | None = Field(
        default=None,
        examples=["TestApplicationId"],
        description="Id of the application in case there is more than one for the AccountId",
    )
    ModelId: str = Field(
        ...,
        examples=["ChurnModelForDrivers"],
        description="User supplied ID of the model trained",
    )

    current_step: NonNegativeInt = Field(
        ...,
        examples=[0],
        description="number of records (rows) ingested",
    )
    current_step_percentage: float = Field(
        ...,
        examples=[0.21],
        description="the percentage of the current step completed",
    )
    total_no_of_steps: NonNegativeInt = Field(
        ...,
        examples=[20],
        description="total number of steps for training the model",
    )


class StartPrediction(BaseModel):
    """Request to start prediction."""

    AccountId: NonNegativeInt = Field(
        ..., examples=[202020], description="ID of an account"
    )
    ApplicationId: str | None = Field(
        default=None,
        examples=["TestApplicationId"],
        description="Id of the application in case there is more than one for the AccountId",
    )
    ModelId: str = Field(
        ...,
        examples=["ChurnModelForDrivers"],
        description="User supplied ID of the model trained",
    )

    task_type: TaskType = Field(
        ...,
        examples=["churn"],
        description="Name of the model used (churn, propensity to buy)",
    )


class ModelMetrics(BaseModel):
    """The standard metrics for classification models.

    The most important metrics is AUC for unbalanced classes such as churn. Metrics such as
    accuracy are not very useful since they are easily maximized by outputting the most common
    class all the time.
    """

    AccountId: NonNegativeInt = Field(
        ..., examples=[202020], description="ID of an account"
    )
    ApplicationId: str | None = Field(
        default=None,
        examples=["TestApplicationId"],
        description="Id of the application in case there is more than one for the AccountId",
    )
    ModelId: str = Field(
        ...,
        examples=["ChurnModelForDrivers"],
        description="User supplied ID of the model trained",
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        examples=["2021-03-28T00:34:08"],
        description="UTC time when the model was trained",
    )
    task_type: TaskType = Field(
        ...,
        examples=["churn"],
        description="Name of the model used (churn, propensity to buy)",
    )

    auc: float = Field(
        ..., examples=[0.91], description="Area under ROC curve", ge=0.0, le=1.0
    )
    f1: float = Field(..., examples=[0.89], description="F-1 score", ge=0.0, le=1.0)
    precission: float = Field(
        ..., examples=[0.84], description="precission", ge=0.0, le=1.0
    )
    recall: float = Field(..., examples=[0.82], description="recall", ge=0.0, le=1.0)
    accuracy: float = Field(
        ..., examples=[0.82], description="accuracy", ge=0.0, le=1.0
    )

    class Config:
        json_encoders = {  # noqa: RUF012
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S"),
        }


class Prediction(BaseModel):
    """Prediction."""

    AccountId: NonNegativeInt = Field(
        ..., examples=[202020], description="ID of an account"
    )
    ApplicationId: str | None = Field(
        default=None,
        examples=["TestApplicationId"],
        description="Id of the application in case there is more than one for the AccountId",
    )
    ModelId: str = Field(
        ...,
        examples=["ChurnModelForDrivers"],
        description="User supplied ID of the model trained",
    )

    PersonId: NonNegativeInt = Field(
        ..., examples=[12345678], description="ID of a person"
    )
    prediction_time: datetime = Field(
        default_factory=datetime.now,
        examples=["2021-03-28T00:34:08"],
        description="UTC time of prediction",
    )
    task_type: TaskType = Field(
        ...,
        examples=["churn"],
        description="Name of the model used (churn, propensity to buy)",
    )
    score: float = Field(
        ...,
        examples=[0.4321],
        description="Prediction score (e.g. the probability of churn in the next 28 days)",
        ge=0.0,
        le=1.0,
    )

    class Config:
        json_encoders = {  # noqa: RUF012
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S"),
        }
