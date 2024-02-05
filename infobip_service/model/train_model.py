from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from infobip_service.dataset.load_dataset import UserHistoryDataset

def get_dataloaders(processed_data_path: Path, definition_id_vocabulary: List[str], time_stats: Dict[str, float], * , batch_size: int = 16, num_workers: int = 8, pin_memory: bool = True):
    return {
        name: DataLoader(
            UserHistoryDataset(processed_data_path / f"{name}.parquet", definition_id_vocabulary=definition_id_vocabulary, time_mean=time_stats["mean"]),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ) for name in ["train", "validation", "test"]
    }

def train_epoch(model, dataloader, optimizer, loss_fn, device) -> None:
    model.train()
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

def evaluate_loss(model, dataloader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, dataloaders, optimizer, loss_fn, device, num_epochs: int = 10):
    print(f"Start Loss: {evaluate_loss(model, dataloaders['validation'], loss_fn, device)}")
    for epoch in range(num_epochs):
        train_epoch(model, dataloaders["train"], optimizer, loss_fn, device)
        validation_loss = evaluate_loss(model, dataloaders["validation"], loss_fn, device)
        print(f"Epoch {epoch + 1}, validation loss: {validation_loss:.3f}")
    return model