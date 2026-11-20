"""
Executes the main training run
"""

import logging

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchinfo import summary
from pathlib import Path

from new_model_architecture.settings import settings
from new_model_architecture.config import TRAINING_DEFAULTS
from new_model_architecture.data_preprocessing import load_dataloader_v1
from new_model_architecture.model import GNN
from new_model_architecture.train import train_one_epoch, validate_one_epoch
from new_model_architecture.utils import checkpoint_model, configure_logging, set_training_folder




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

logger.info(summary(model))

logger.info("Training Model...")
logger.info(f"Epochs: {TRAINING_DEFAULTS['NUM_EPOCHS']}, Batch Size: {TRAINING_DEFAULTS['BATCH_SIZE']}, Learning Rate: {TRAINING_DEFAULTS['LEARNING_RATE']}, Weight Decay: {TRAINING_DEFAULTS['WEIGHT_DECAY']}")


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

    logger.info(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
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
        logger.info(f"Checkpoint saved at epoch {epoch} with validation loss: {val_loss:.4f}")
    

    print(f"Epoch {epoch}, Loss: {avg_loss}")

save_model_path = "new_model_architecture/checkpoint/final_model.pth"
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")
