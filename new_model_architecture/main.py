"""
Executes the main training run
"""

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from new_model_architecture.config import TRAINING_DEFAULTS
from new_model_architecture.data.dataloader import (
    SionnaMultiSceneDataset,
    SnapshotDataset,
)
from new_model_architecture.model import GNN
from new_model_architecture.train import train_one_epoch, validate_one_epoch
from new_model_architecture.utils import checkpoint_model
from torch.optim.lr_scheduler import OneCycleLR
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset: SionnaMultiSceneDataset = SionnaMultiSceneDataset(
    filepaths=[
        "new_model_architecture/data/multi_scene_ds/munich_train.h5",
        "new_model_architecture/data/multi_scene_ds/florence_train.h5"
    ])

# Split before expanding into snapshots
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
    generator=torch.Generator().manual_seed(42)
)

train_snapshot_dataset: SnapshotDataset = SnapshotDataset(train_dataset)
val_snapshot_dataset: SnapshotDataset = SnapshotDataset(val_dataset)



train_dataloader = DataLoader(
    dataset=train_snapshot_dataset,
    batch_size=TRAINING_DEFAULTS["BATCH_SIZE"],
    shuffle=True,
    num_workers=0,
)
val_dataloader = DataLoader(
    dataset=val_snapshot_dataset,
    batch_size=TRAINING_DEFAULTS["BATCH_SIZE"],
    shuffle=False,
    num_workers=0,
)

print(f"num_batches: {len(train_dataloader)}")
print(f"num_val_batches: {len(val_dataloader)}")

model = GNN()
model = model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAINING_DEFAULTS["LEARNING_RATE"],
    weight_decay=TRAINING_DEFAULTS["WEIGHT_DECAY"],
)
criterion = nn.MSELoss()
scheduler = OneCycleLR(
    optimizer, 
    max_lr=TRAINING_DEFAULTS["LEARNING_RATE"], 
    epochs=TRAINING_DEFAULTS["NUM_EPOCHS"],
    steps_per_epoch=len(train_dataloader),
    pct_start=0.2, 
    anneal_strategy='linear'
)

print(summary(model))

print("Training model...")
print(f"Epochs: {TRAINING_DEFAULTS['NUM_EPOCHS']}, Batch Size: {TRAINING_DEFAULTS['BATCH_SIZE']}, Learning Rate: {TRAINING_DEFAULTS['LEARNING_RATE']}, Weight Decay: {TRAINING_DEFAULTS['WEIGHT_DECAY']}")


lowest_val_loss = float('inf')
# Training loop
for epoch in range(TRAINING_DEFAULTS["NUM_EPOCHS"]):
    avg_loss = train_one_epoch(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,    
        criterion=criterion,
        device=device
    )
    val_loss: float = validate_one_epoch(
        model=model,
        val_loader=val_dataloader,
        criterion=criterion,
        device=device
    )

    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        checkpoint_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=val_loss,
            checkpoint_path="new_model_architecture/checkpoint/best_model.pth"
        )
        print(f"Checkpoint saved at epoch {epoch} with validation loss: {val_loss:.4f}")

    print(f"Epoch {epoch}, Loss: {avg_loss}")

save_model_path = "new_model_architecture/checkpoint/final_model.pth"
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")