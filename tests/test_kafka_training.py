from datetime import datetime
from pathlib import Path


def test_get_paths(patch_envs):
    from infobip_service.kafka_training import get_paths

    downloading_root_path = Path("downloading_test_path")
    training_root_path = Path("training_test_path")
    AccountId = 317238  # noqa: N806
    ApplicationId = ""  # noqa: N806
    ModelId = "20060"  # noqa: N806
    dt = datetime.now().date().isoformat()

    actual = get_paths(
        downloading_root_path=downloading_root_path,
        training_root_path=training_root_path,
        AccountId=AccountId,
        ApplicationId=ApplicationId,
        ModelId=ModelId,
    )

    expected = {
        "input_data_path": Path(
            f"{downloading_root_path}/AccountId-{AccountId}/ApplicationId-{ApplicationId}/ModelId-{ModelId}/{dt}"
        ),
        "preprocessing_path": Path(
            f"{training_root_path}/AccountId-{AccountId}/ApplicationId-{ApplicationId}/ModelId-{ModelId}/{dt}/TimeSeriesMaskedPreprocessing"
        ),
        "training_path": Path(
            f"{training_root_path}/AccountId-{AccountId}/ApplicationId-{ApplicationId}/ModelId-{ModelId}/{dt}/TimeSeriesDownstreamDataChurn"
        ),
        "prediction_path": Path(
            f"{training_root_path}/AccountId-{AccountId}/ApplicationId-{ApplicationId}/ModelId-{ModelId}/{dt}/TimeSeriesDownstreamPrediction"
        ),
    }
    assert actual == expected, actual
