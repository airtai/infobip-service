import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from infobip_service.dataset.load_dataset import UserHistoryDataset
from infobip_service.dataset.preprocessing import processed_data_path
from infobip_service.model import ChurnModel, ChurnProbabilityModel  # type: ignore


def graph_hit_rate(
    model_predictions: pd.DataFrame,
    slices: list[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],  # noqa
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
