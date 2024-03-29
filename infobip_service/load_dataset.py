from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

buckets = [1, 3, 7, 14, 28]

timedelta_buckets = np.array(
    [timedelta(days=days) for days in buckets], dtype="timedelta64[ms]"
)


def _bin_timedelta(
    timedelta: timedelta,
    *,
    timedelta_buckets: np.ndarray = timedelta_buckets,  # type: ignore
) -> int:
    for class_value, timedelta_key in enumerate(timedelta_buckets):
        if timedelta < timedelta_key:
            return class_value

    return len(timedelta_buckets)


def embed_vocab(x: str, vocab: list[str]) -> int:
    try:
        return vocab.index(x)
    except (ValueError, TypeError):
        return len(vocab)


def bin_next_event_user_history(
    user_history: datetime | None,
    *,
    t0: datetime,
    timedelta_buckets: np.ndarray = timedelta_buckets,  # type: ignore
) -> int:
    if user_history is None:
        return len(timedelta_buckets)
    return _bin_timedelta(user_history - t0, timedelta_buckets=timedelta_buckets)


class UserHistoryDataset(Dataset):  # type: ignore
    """Dataset for user histories."""

    def __init__(self, histories_path: Path, definitionId_vocab: list[str]):
        """Initialize dataset."""
        self.histories = pd.read_parquet(histories_path).sort_index()
        self.sample_indexes = list(
            {  # noqa
                index
                for index in self.histories.index.get_level_values("PersonId").to_list()
            }
        )
        self.definitionId_vocab = definitionId_vocab

    def __len__(self) -> int:
        return len(self.sample_indexes)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        actions = self.histories.query(
            f'PersonId == "{self.sample_indexes[idx]}" & level_1 == "DefinitionId"'
        )
        times = self.histories.query(
            f'PersonId == "{self.sample_indexes[idx]}" & level_1 == "OccurredTime"'
        ).copy()
        times[times.columns] = times[times.columns].apply(pd.to_datetime)

        historic_actions = np.apply_along_axis(  # type: ignore
            lambda x: embed_vocab(x, self.definitionId_vocab),  # type: ignore
            0,
            actions.loc[:, actions.columns != "NextEvent"].to_numpy(),
        )
        next_action = np.apply_along_axis(
            lambda x: embed_vocab(x, self.definitionId_vocab),  # type: ignore
            0,
            actions.loc[:, actions.columns == "NextEvent"].to_numpy(),
        )[0]

        historic_times = times.loc[:, times.columns != "NextEvent"].to_numpy()[0]
        next_time = times.loc[:, times.columns == "NextEvent"].to_numpy()[0][0]

        x = np.stack(
            [historic_actions, historic_times.astype("datetime64[s]").astype("int")],
            axis=-1,
        )
        y = bin_next_event_user_history(
            None if next_action == len(self.definitionId_vocab) else next_time,
            t0=historic_times[-1],
        )

        return x, y
