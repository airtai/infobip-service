import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from infobip_service.dataset.load_dataset import (
    UserHistoryDataset,
)
from infobip_service.dataset.preprocessing import processed_data_path