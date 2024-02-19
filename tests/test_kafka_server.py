import logging
from datetime import datetime, timedelta
from time import sleep

from pydantic import BaseModel

from infobip_service.kafka_server import (
    EventData,
    LogMessage,
    ModelMetrics,
    ModelTrainingRequest,
    Prediction,
    StartPrediction,
    Tracker,
    TrainingDataStatus,
    TrainingModelStart,
    TrainingModelStatus,
    get_key,
)


def test_log_message():
    class SomeMessage(BaseModel):
        a: int = 12
        b: str = "hello"

    original_message = SomeMessage()
    msg = LogMessage(
        level=logging.INFO,
        timestamp="2021-03-28T00:34:08",
        message="something went wrong",
        original_message=original_message,
    )

    actual = msg.json(indent=None)
    expected = '{"level": 20, "timestamp": "2021-03-28T00:34:08", "message": "something went wrong"}'
    assert actual == expected, actual

    msg = LogMessage(
        message="something went wrong",
        original_message=original_message,
    )
    assert msg.timestamp is not None, msg.timestamp
    assert datetime.now() - msg.timestamp < timedelta(seconds=1), (
        datetime.now() - msg.timestamp
    )


def test_model_training_request():
    model_training_request = ModelTrainingRequest(
        AccountId=12345,
        ModelId="10001",
        OccurredTime="2021-03-28T00:34:08",
        task_type="churn",
        total_no_of_records=1000,
    )

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "10001", "task_type": "churn", "total_no_of_records": 1000}'
    actual = model_training_request.json()

    assert actual == expected, actual

    parsed = ModelTrainingRequest.parse_raw(actual)
    assert parsed == model_training_request


def test_event_data():
    event_data = EventData(
        AccountId=12345,
        ModelId="10001",
        DefinitionId="BigButton",
        PersonId=123456789,
        OccurredTime="2021-03-28T00:34:08",
        OccurredTimeTicks=1616891648496,
    )

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "10001", "DefinitionId": "BigButton", "OccurredTime": "2021-03-28T00:34:08", "OccurredTimeTicks": 1616891648496, "PersonId": 123456789}'
    actual = event_data.json()

    assert actual == expected, actual

    parsed = EventData.parse_raw(actual)
    assert parsed == event_data


def test_training_data_status():
    training_data_status = TrainingDataStatus(
        AccountId=12345,
        ModelId="10001",
        no_of_records=23,
        total_no_of_records=54,
    )

    expected = '{"AccountId": 12345, "ModelId": "10001", "no_of_records": 23, "total_no_of_records": 54}'
    actual = training_data_status.json()

    assert actual == expected, actual

    parsed = TrainingDataStatus.parse_raw(actual)
    assert parsed == training_data_status


def test_training_model_start():
    training_model_start = TrainingModelStart(
        AccountId=12345,
        ModelId="10001",
        task_type="churn",
        no_of_records=100,
    )

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "10001", "task_type": "churn", "no_of_records": 100}'
    actual = training_model_start.json()

    assert actual == expected, actual

    parsed = TrainingModelStart.parse_raw(actual)
    assert parsed == training_model_start


def test_get_key():
    training_model_start = TrainingModelStart(
        AccountId=12345,
        ModelId="10001",
        task_type="churn",
        no_of_records=100,
    )

    actual = get_key(training_model_start)
    assert actual == b"AccountId='12345', ModelId='10001'", actual

    actual = get_key(training_model_start, ["task_type"])
    assert actual == b"task_type='churn'", actual


def test_tracker_1():
    tracker = Tracker(limit=10, timeout=5, abort_after=10)

    assert not tracker.finished()

    assert tracker.update(9)
    assert not tracker.update(9)

    assert not tracker.finished()

    assert tracker.update(10)

    assert tracker.finished()


def test_tracker_2():
    tracker = Tracker(limit=10, timeout=1, abort_after=10)

    assert not tracker.finished()

    tracker.update(9)

    assert not tracker.finished()

    sleep(1.1)

    assert tracker.finished()


def test_tracker_3():
    tracker = Tracker(limit=10, timeout=1, abort_after=2)
    sleep(1.1)

    assert not tracker.finished()
    assert not tracker.aborted()

    sleep(1.1)

    assert tracker.finished()
    assert tracker.aborted()


def test_training_model_status():
    training_model_status = TrainingModelStatus(
        AccountId=12345,
        ModelId="123",
        current_step=1,
        current_step_percentage=0.21,
        total_no_of_steps=20,
    )

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "123", "current_step": 1, "current_step_percentage": 0.21, "total_no_of_steps": 20}'
    actual = training_model_status.json()

    assert actual == expected, actual

    parsed = TrainingModelStatus.parse_raw(actual)
    assert parsed == training_model_status


def test_start_prediction():
    model_metrics = StartPrediction(AccountId=12345, ModelId="10058", task_type="churn")

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "10058", "task_type": "churn"}'
    actual = model_metrics.json()

    assert actual == expected, actual


def test_model_metrics():
    model_metrics = ModelMetrics(
        AccountId=12345,
        ModelId="456",
        timestamp="2021-03-28T00:34:08.123456",
        task_type="churn",
        auc=0.95,
        recall=0.94,
        precision=0.98,
        accuracy=0.99,
        f1=2 * 0.94 * 0.98 / (0.94 + 0.98),
    )

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "456", "timestamp": "2021-03-28T00:34:08", "task_type": "churn", "auc": 0.95, "f1": 0.9595833333333332, "precision": 0.98, "recall": 0.94, "accuracy": 0.99}'
    actual = model_metrics.json()

    assert actual == expected, actual

    model_metrics = ModelMetrics(
        AccountId=12345,
        ModelId="456",
        timestamp="2021-03-28T00:34:08",
        task_type="churn",
        auc=0.95,
        recall=0.94,
        precision=0.98,
        accuracy=0.99,
        f1=2 * 0.94 * 0.98 / (0.94 + 0.98),
    )

    parsed = ModelMetrics.parse_raw(actual)
    assert parsed == model_metrics, f"{parsed}!={model_metrics}"


def test_prediction():
    prediction = Prediction(
        AccountId=12345,
        ModelId="20001",
        PersonId=123456789,
        prediction_time="2021-03-28T00:34:08.123456",
        task_type="churn",
        score=0.873,
    )

    expected = '{"AccountId": 12345, "ApplicationId": null, "ModelId": "20001", "PersonId": 123456789, "prediction_time": "2021-03-28T00:34:08", "task_type": "churn", "score": 0.873}'
    actual = prediction.json()

    assert actual == expected, actual

    prediction = Prediction(
        AccountId=12345,
        ModelId="20001",
        PersonId=123456789,
        prediction_time="2021-03-28T00:34:08",
        task_type="churn",
        score=0.873,
    )

    parsed = Prediction.parse_raw(actual)
    assert parsed == prediction, f"{parsed}!={prediction}"
