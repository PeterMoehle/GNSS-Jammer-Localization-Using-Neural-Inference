import logging
from pathlib import Path

import torch
import random

class AverageMeter:
    """
    Class to track average over stream of values for MSE loss during training and testing.

    Attributes:
        val (float): The latest value added to the meter.
        avg (float): The current average of all added values.
        sum (float): The sum of all added values.
        count (int): The number of entries that have been added.
    """
    def __init__(self):
        self.val, self.avg, self.sum, self.count = float('inf'), None, None, None
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def checkpoint_model(model, optimizer, scheduler, epoch, val_loss, dataset_filepaths, checkpoint_path):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to save.
        epoch (int): The current epoch number.
        val_loss (float): The current validation loss value.
        dataset_filepaths (list): List of dataset file paths used for training.
        checkpoint_path (str): Path to save the checkpoint file.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": val_loss,
        "dataset_filepaths": dataset_filepaths,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
    }, checkpoint_path)

def configure_logging(log_file_path):
    """
    Configure logging to log messages to both console and a file.

    Args:
        log_file_path (str): Path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
):
    """
    Load the model checkpoint and restore the model, optimizer, and scheduler states.
    ):
    Load the model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to restore.
        optimizer (torch.optim.Optimizer): The optimizer to restore.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to restore.

    Returns:
        tuple: A tuple containing the restored model, optimizer, scheduler, start epoch, and lowest validation loss.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    lowest_val_loss = checkpoint['loss']

    return checkpoint, model, optimizer, scheduler, start_epoch, lowest_val_loss


def set_training_folder(parent_path: Path)-> Path:
    """
    Create a new training folder with an incremented number in the parent directory.

    Args:
        parent_path (str): The path to the parent directory where the training folder will be created.

    Returns: Child folder where all training related data should be kept.
    """
    parent_path = Path(parent_path)
    parent_path.mkdir(parents=True, exist_ok=True)

    # Find the next available training folder number
    existing_folders = [d for d in parent_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    existing_numbers = [int(d.name.split("_")[1]) for d in existing_folders if d.name.split("_")[1].isdigit()]
    next_number = max(existing_numbers, default=0) + 1

    # Create the new training folder
    new_training_folder = parent_path / f"run_{next_number}"
    new_training_folder.mkdir()

    return new_training_folder