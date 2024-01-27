import json
from datetime import datetime
from pathlib import Path

import dask.dataframe as df
import numpy as np
import pandas as pd
import torch

from infobip_service.load_dataset import embed_vocab
from infobip_service.model import ChurnProbabilityModel  # type: ignore
from infobip_service.preprocessing import _create_user_history, preprocess_dataset
from infobip_service.train_model import train_model


def prepare_user_history(
    user_history: pd.DataFrame, definitionId_vocab: list[str]
) -> pd.DataFrame:
    actions = user_history.loc[["DefinitionId"]]

    actions = np.apply_along_axis(  # type: ignore
        lambda x: embed_vocab(x, definitionId_vocab),  # type: ignore
        0,
        actions.to_numpy(),
    )

    times = user_history.loc[["OccurredTime"]]

    times = times[times.columns].apply(pd.to_datetime).to_numpy()[0]

    return torch.Tensor(
        np.stack(
            [actions, times.astype("datetime64[s]").astype("int")],
            axis=-1,
        )
    ).long()


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
        ).cpu()
        return self

    def predict(
        self,
        prediction_inputs_path: Path,
        t0: datetime = datetime.now(),  # noqa
    ) -> ChurnProbabilityModel:
        with Path.open(self.processed_data_path / "DefinitionId_vocab.json", "rb") as f:
            vocab = json.load(f)

        if self.model is None:
            raise ValueError("Model is not trained")

        prediction_model = ChurnProbabilityModel(self.model)  # type: ignore

        prediction_inputs = df.read_parquet(prediction_inputs_path)

        users = prediction_inputs.index.unique().compute()

        churn_predictions = None

        for user in users:
            user_history_raw = prediction_inputs.loc[user].compute()
            user_history_raw["OccurredTime"] = pd.to_datetime(
                user_history_raw["OccurredTime"]
            )
            user_history = prepare_user_history(
                _create_user_history(user_history_raw, t0=t0, history_size=64),
                definitionId_vocab=vocab,
            )

            prediction = pd.DataFrame(
                {
                    "prediction": [prediction_model(user_history, t0).item()],
                },
                index=pd.Index([user], name="PersonId"),
            )

            if churn_predictions is None:
                churn_predictions = prediction
            else:
                churn_predictions = pd.concat([churn_predictions, prediction])

        return churn_predictions
