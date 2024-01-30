import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from infobip_service.dataset.load_dataset import UserHistoryDataset
from infobip_service.dataset.preprocessing import processed_data_path
from infobip_service.model import ChurnModel  # type: ignore

model_path = Path() / ".." / "models" / "model.pt"


def _run_training_loop(
    model: ChurnModel,
    train_dataset: DataLoader,  # type: ignore
    validation_dataset: DataLoader,  # type: ignore
    epochs: int,
    learning_rate: float,
    device: str,
) -> torch.nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.NLLLoss()
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_dataset):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(validation_dataset):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                total_loss += loss.item()

        average_loss = total_loss / len(validation_dataset)
        print(f"Epoch: {epoch}, Average Validation Loss: {average_loss}")  # noqa

    return model


def train_model(
    processed_data_path: Path, epochs: int = 2, learning_rate: float = 0.001
) -> torch.nn.Module:
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    with Path.open(processed_data_path / "time_stats.json", "rb") as f:
        time_stats = json.load(f)

    datasets = {
        k: DataLoader(
            UserHistoryDataset(
                processed_data_path / f"{k}.parquet", definitionId_vocab=vocab
            ),
            batch_size=16,
            num_workers=32,
            pin_memory=True,
        )
        for k in ["train", "validation", "test"]
    }

    model = ChurnModel(
        definition_id_vocab_size=len(vocab) + 1,
        time_normalization_params=time_stats,
        embedding_dim=10,
        churn_bucket_size=6,
    )

    trained_model = _run_training_loop(
        model,
        datasets["train"],
        datasets["validation"],
        epochs=epochs,
        learning_rate=learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return trained_model


if __name__ == "__main__":
    model = train_model(processed_data_path)
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
