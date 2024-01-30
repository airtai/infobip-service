import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from infobip_service.load_dataset import UserHistoryDataset
from infobip_service.model import ChurnModel, ChurnProbabilityModel  # type: ignore
from infobip_service.preprocessing import processed_data_path
from infobip_service.train_model import model_path

calculated_predictions_path = processed_data_path / "predictions.parquet"


def calculate_predictions(
    *,
    model_path: Path,
    processed_data_path: Path,
    user_history_path: Path,
    calculated_predictions_path: Path,
    max_time: datetime,
) -> None:
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    with Path.open(processed_data_path / "time_stats.json", "rb") as f:
        time_stats = json.load(f)

    model = ChurnModel(
        definition_id_vocab_size=len(vocab) + 1,
        time_normalization_params=time_stats,
        embedding_dim=10,
        churn_bucket_size=6,
    )

    model.load_state_dict(torch.load(model_path))

    model = ChurnProbabilityModel(model)  # type: ignore

    test_dataset = DataLoader(
        UserHistoryDataset(user_history_path, definitionId_vocab=vocab),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    model_predictions = pd.DataFrame(columns=["user_churned", "churn_probability"])

    for sample, user_churned in tqdm.tqdm(test_dataset):
        churn_probability = model(sample, max_time=max_time)
        model_predictions = pd.concat(
            [
                model_predictions,
                pd.DataFrame(
                    {
                        "user_churned": user_churned.item() == 5,
                        "churn_probability": churn_probability.item(),
                    },
                    index=[0],
                ),
            ]
        )

    model_predictions.to_parquet(calculated_predictions_path)


def graph_hit_rate(
    model_predictions: pd.DataFrame,
    slices: list[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
) -> None:
    model_predictions.sort_values("churn_probability", inplace=True)

    pct_churned = len(model_predictions[model_predictions["user_churned"]]) / len(
        model_predictions
    )

    hit_rates = []
    for slice in slices:
        predictions = model_predictions.head(int(len(model_predictions) * slice))
        hit_rates.append(
            len(predictions[predictions["user_churned"]]) / len(predictions)
        )

    plt.plot(slices, hit_rates)
    plt.axhline(pct_churned, color="red", linestyle="--")
    plt.axvline(pct_churned, color="red", linestyle="--")
    plt.xlabel("slice percentage")
    plt.xscale("log")
    plt.ylabel("hit rate")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    calculate_predictions(
        model_path=model_path,
        processed_data_path=processed_data_path,
        user_history_path=processed_data_path / "test_prepared.parquet",
        calculated_predictions_path=calculated_predictions_path,
        max_time=datetime.strptime("2023-12-23 00:00:30.906000", "%Y-%m-%d %H:%M:%S.%f")
        - timedelta(days=28),
    )
