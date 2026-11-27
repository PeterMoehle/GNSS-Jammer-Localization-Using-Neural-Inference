"""
Executes the main training run
"""

import logging
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchinfo import summary
from torch_geometric.utils import degree

from new_model_architecture.data_preprocessing import load_dataloader_v1
from new_model_architecture.model import GNN
from new_model_architecture.settings import settings
from new_model_architecture.train import train_one_epoch, validate_one_epoch
from new_model_architecture.utils import (
    checkpoint_model,
    configure_logging,
    load_checkpoint,
    set_training_folder,
)

training_folder: Path = set_training_folder(Path(settings.RUN_PATH)) 


configure_logging(log_file_path=training_folder / "training.log")

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

train_dataset_filepaths: list[str | Path] = [
    "new_model_architecture/data/multi_scene_ds/munich_train.h5",
    "new_model_architecture/data/multi_scene_ds/florence_train.h5"
]
test_dataset_filepaths: list[str | Path] = [
    "new_model_architecture/data/multi_scene_ds/etoile_test.h5"
]

train_dataloader, val_dataloader, test_dataloader = load_dataloader_v1(
    train_dataset_filepaths=train_dataset_filepaths,
    test_dataset_filepaths=test_dataset_filepaths
)


logger.info(f"num_batches: {len(train_dataloader)}")
logger.info(f"num_val_batches: {len(val_dataloader)}")

pna_degree_histogram = None
if settings.MODEL_TYPE.upper() == "PNA":
    sample_batch = next(iter(train_dataloader), None)
    if sample_batch is None:
        raise ValueError("Cannot initialize PNA with an empty training dataloader.")
    in_degrees = degree(
        sample_batch.edge_index[1],
        num_nodes=sample_batch.num_nodes,
        dtype=torch.long,
    )
    pna_degree_histogram = torch.bincount(in_degrees)

model = GNN(deg=pna_degree_histogram)
model = model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=settings.LEARNING_RATE,
    weight_decay=settings.WEIGHT_DECAY,
)
criterion = nn.MSELoss()
scheduler = OneCycleLR(
    optimizer, 
    max_lr=settings.LEARNING_RATE, 
    epochs=settings.NUM_EPOCHS,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.2, 
    anneal_strategy='linear'
)

logger.info(summary(model, verbose=0, device=device.type))

logger.info("Training Model")
logger.info(f"Epochs: {settings.NUM_EPOCHS}, Batch Size: {settings.BATCH_SIZE}, Learning Rate: {settings.LEARNING_RATE}, Weight Decay: {settings.WEIGHT_DECAY}")

lowest_val_loss = float('inf')
start_epoch = 0  # Initialize start_epoch to 0
patience_counter = 0  # Initialize patience counter for early stopping

# Training loop
if settings.CONTINUE_TRAINING:
    if not settings.CHECKPOINT_PATH:
        logger.error("CONTINUE_TRAINING is set to True, but CHECKPOINT_PATH is not provided.")
        raise ValueError("CHECKPOINT_PATH must be provided when CONTINUE_TRAINING is True.")


    checkpoint_path = Path(settings.CHECKPOINT_PATH)
    if checkpoint_path.exists():
        checkpoint, model, optimizer, scheduler, start_epoch, lowest_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

        random.setstate(checkpoint["python_random_state"])
        torch.random.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

        if checkpoint["dataset_filepaths"] != train_dataset_filepaths:
            raise ValueError("Training dataset differs from checkpoint dataset")

        logger.info(f"Resumed from epoch {start_epoch} with validation loss: {lowest_val_loss:.4f}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        start_epoch = 0


for epoch in range(start_epoch, settings.NUM_EPOCHS):
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

    logger.info(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
    if val_loss < lowest_val_loss:
        patience_counter = 0
        lowest_val_loss = val_loss
        checkpoint_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_loss,
            dataset_filepaths=train_dataset_filepaths,
            checkpoint_path=training_folder / "best_model.pth"
        )
        logger.info(f"Checkpoint saved at epoch {epoch} with validation loss: {val_loss:.4f}")
    else:
        logger.info(f"No improvement in validation loss. Current: {val_loss:.4f}, Best: {lowest_val_loss:.4f}")
        patience_counter += 1
        if patience_counter >= settings.PATIENCE:
            logger.info(f"Early stopping triggered after {settings.PATIENCE} epochs without improvement.")
            break

save_model_path = training_folder / "final_model.pth"
torch.save(model.state_dict(), save_model_path)
logger.info(f"Final model saved at: {save_model_path}")
