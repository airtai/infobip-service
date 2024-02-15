from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

bins = [
    np.timedelta64(days, "D").astype("timedelta64[s]").astype(int)
    for days in [0, 1, 3, 5, 7, 14, 28]
]


def prepare_sample(
    df: pd.DataFrame, *, definition_id_vocabulary: list[str], time_mean: int
) -> np.ndarray:
    has_history = df["HasHistory"].values
    actions = df["DefinitionId"].values
    embedded_actions = np.ones_like(actions, dtype=int) * len(definition_id_vocabulary)

    for index, definition in enumerate(definition_id_vocabulary):
        embedded_actions[has_history] = np.where(
            definition == actions[has_history], index, embedded_actions[has_history]
        )

    occurred_times = np.where(
        has_history,
        df["OccurredTime"].values.astype("datetime64[s]").astype("int"),
        time_mean,
    )

    return np.stack([embedded_actions, occurred_times], axis=1)


class UserHistoryDataset(Dataset):  # type: ignore
    """Dataset for user histories."""

    def __init__(
        self,
        histories_path: Path,
        *,
        definition_id_vocabulary: list[str],
        time_mean: int,
        bins: list[np.timedelta64] = bins,
        history_size: int = 32,
    ):
        """Initialize dataset."""
        self.histories = pd.read_parquet(histories_path)
        self.definition_id_vocabulary = definition_id_vocabulary
        self.time_mean = time_mean
        self.history_size = history_size
        self.bins = bins

    def __len__(self) -> int:
        return len(self.histories) // self.history_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        history = self.histories.iloc[
            idx * self.history_size : (idx + 1) * self.history_size
        ]
        x = prepare_sample(
            history,
            definition_id_vocabulary=self.definition_id_vocabulary,
            time_mean=self.time_mean,
        )
        y = (
            np.digitize(
                history.iloc[-1]["OccurredTimeDelta"].total_seconds(),
                right=False,
                bins=self.bins,
            )
            - 1
        )
        return x, y
