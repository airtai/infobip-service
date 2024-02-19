from datetime import datetime
from pathlib import Path

import pandas as pd


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
        # TODO: Implement preprocessing and training
        return self

    def predict(
        self,
        prediction_inputs_path: Path,
        t0: datetime = datetime.now(),  # noqa
    ) -> pd.DataFrame:
        ## TODO: prepare samples, return model predictions
        pass
