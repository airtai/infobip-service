import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from infobip_service.load_dataset import UserHistoryDataset
from infobip_service.model import ChurnModel  # type: ignore
from infobip_service.preprocessing import processed_data_path


def train_model(
    model: ChurnModel,
    train_dataset: DataLoader,  # type: ignore
    validation_dataset: DataLoader,  # type: ignore
    epochs: int,
    learning_rate: float,
    device: str,
) -> torch.nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
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

        print("Validating...")
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
        print(f"Epoch: {epoch}, Average Validation Loss: {average_loss}")

    return model


if __name__ == "__main__":
    with Path.open(processed_data_path / "DefinitionId_vocab.json", "rb") as f:
        vocab = json.load(f)

    with Path.open(processed_data_path / "time_stats.json", "rb") as f:
        time_stats = json.load(f)

    print("Loading datasets...")
    datasets = {
        k: DataLoader(
            UserHistoryDataset(
                processed_data_path / f"{k}_prepared.parquet", definitionId_vocab=vocab
            ),
            batch_size=16,
            num_workers=32,
            pin_memory=True,
        )
        for k in ["train", "validation", "test"]
    }
    print("Done.")

    model = ChurnModel(
        definition_id_vocab_size=len(vocab) + 1,
        time_normalization_params=time_stats,
        embedding_dim=10,
        churn_bucket_size=6,
    )

    print("Training model...")
    trained_model = train_model(
        model,
        datasets["train"],
        datasets["validation"],
        epochs=2,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("Done.")
