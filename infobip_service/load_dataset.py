from datetime import datetime, timedelta
from typing import List, Optional

# import pandas as pd

# from infobip_service.preprocessing import processed_data_path

timedelta_buckets = [timedelta(days=days) for days in [1, 3, 7, 14, 28]]


def _bin_timedelta(
    timedelta: timedelta, *, timedelta_buckets: List[timedelta] = timedelta_buckets
) -> int:
    for class_value, timedelta_key in enumerate(timedelta_buckets):
        if timedelta < timedelta_key:
            return class_value

    return len(timedelta_buckets)


def bin_next_event_user_history(
    user_history: Optional[datetime],
    *,
    t0: datetime,
    timedelta_buckets: List[timedelta] = timedelta_buckets,
) -> int:
    if user_history is None:
        return len(timedelta_buckets)
    return _bin_timedelta(user_history - t0, timedelta_buckets=timedelta_buckets)


# def load_dataset(
#     processed_data_path: Path = processed_data_path,
# ) -> Tuple[ListDataset, ListDataset, ListDataset]:
#     train_set = pd.read_parquet(processed_data_path / "train_prepared.parquet")
#     validation_set = pd.read_parquet(
#         processed_data_path / "validation_prepared.parquet"
#     )
#     test_set = pd.read_parquet(processed_data_path / "test_prepared.parquet")
