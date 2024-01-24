import json

import pytest
from torch.utils.data import DataLoader

from infobip_service.model import ChurnModel
from infobip_service.load_dataset import UserHistoryDataset
from infobip_service.preprocessing import processed_data_path


@pytest.mark.skip(reason="Dataset not available on CI/CD")
def test_model_forward():
    with open(processed_data_path/"DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    with open(processed_data_path/"time_stats.json", "rb") as f:
        time_stats = json.load(f)

    dataset = DataLoader(UserHistoryDataset(processed_data_path/"validation_prepared.parquet", definitionId_vocab=vocab), batch_size=4, pin_memory=True)

    model = ChurnModel(definition_id_vocab_size=len(vocab)+1, time_normalization_params=time_stats, embedding_dim=10, churn_bucket_size=6)

    x, _ = next(iter(dataset))

    assert model(x).shape == (4, 1, 6)
