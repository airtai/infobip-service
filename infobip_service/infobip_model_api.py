from datetime import datetime
from pathlib import Path

from infobip_service.load_dataset import UserHistoryDataset
from infobip_service.model import ChurnProbabilityModel  # type: ignore
from infobip_service.preprocessing import preprocess_dataset
from infobip_service.train_model import train_model


class TimeSeriesDownstreamSolution:
    """Time series downstream solution."""

    def __init__(
        self,
        raw_data_path: Path,
        processed_data_path: Path,
        epochs: int,
        learning_rate: float,
    ):
        """Initialize downstream solution."""
        self.raw_data_path = raw_data_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.processed_data_path = processed_data_path
        self.model = None

    def train(self) -> "TimeSeriesDownstreamSolution":
        preprocess_dataset(self.raw_data_path, self.processed_data_path)
        self.model = train_model(  # type: ignore
            self.processed_data_path, self.epochs, self.learning_rate
        )
        return self

    def predict(
        self, t0: datetime, churn_prediction_data: UserHistoryDataset
    ) -> ChurnProbabilityModel:
        if self.model is None:
            raise ValueError("Model is not trained")

        prediction_model = ChurnProbabilityModel(self.model)  # type: ignore

        churn_probability = [
            prediction_model(churn_prediction_data, t0)
            for churn_prediction_data in churn_prediction_data
        ]

        return churn_probability
