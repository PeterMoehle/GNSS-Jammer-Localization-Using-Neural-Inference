import csv
import logging
import os
import random
import statistics

import numpy as np
import torch

from custom_logging import setup_logging
from global_config import global_config

# Setup custom logging
setup_logging()

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
        self.val, self.avg, self.sum, self.count = None, None, None, None
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


def save_model_predictions(seed, data_class, predictions, actuals, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list):
    """
    Appends model predictions and associated metadata to a CSV file.

    Args:
        seed (int): Random seed used to ensure reproducibility.
        data_class (str): The class of data used.
        predictions (list): List of predicted values from the model.
        actuals (list): List of actual values.
        perc_completion_list (list): List of percentages indicating the completion of trajectory.
        pl_exp_list (list): List of path loss exponent values associated with predictions.
        sigma_list (list): List of sigma values indicating noise level variations.
        jtx_list (list): List of jammer transmit power values.
        num_samples_list (list): List of the number of samples considered in each prediction instance.
        weight_list (list): List of weight values affecting the prediction or evaluation.

    """
    # Save predictions, actuals, and perc_completion to a CSV file
    if global_config.args.inference:
        file = f'{global_config.args.experiments_folder}predictions_{global_config.args.model}_{data_class}_inference.csv'
    else:
        file = f'{global_config.args.experiments_folder}predictions_{global_config.args.model}_{data_class}.csv'

    # Check if the file exists
    file_exists = os.path.isfile(file)
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)

        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(['Seed', 'Prediction', 'Actual', 'Percentage Completion', 'PL', 'Sigma', 'JTx', 'Number Samples', 'Weight'])

        # Write the prediction, actual, and percentage completion data
        for pred, act, perc, plexp, sigma, jtx, numsamples, weight in zip(predictions, actuals, perc_completion_list, pl_exp_list, sigma_list,
                                                                          jtx_list, num_samples_list, weight_list):
            writer.writerow([seed, pred, act, perc, plexp, sigma, jtx, numsamples, weight])

    logging.info(f"Saved seed {seed} test set predictions to {file}")


def save_rmse_mae_stats(rmse_vals, mae_vals, data_class):
    """
    Saves Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) statistics to CSV.

    Args:
        rmse_vals (list[float]): List of RMSE values from model evaluations.
        mae_vals (list[float]): List of MAE values from model evaluations.
        data_class (str): Category of the data used, influences the row description in the CSV.

    """
    if len(rmse_vals) > 1:
        mean_rmse = statistics.mean(rmse_vals)
        std_rmse = statistics.stdev(rmse_vals)
        mean_mae = statistics.mean(mae_vals)
        std_mae = np.std(mae_vals)
    else:
        mean_rmse = statistics.mean(rmse_vals) if rmse_vals else float('nan')
        std_rmse = float('nan')
        mean_mae = statistics.mean(mae_vals) if mae_vals else float('nan')
        std_mae = float('nan')

    # Saving RMSE and MAE statistics
    csv_file_path = global_config.args.experiments_folder + 'metrics_statistics.csv'
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Data Class', 'RMSE', 'MAE'])  # Adding column headers
        writer.writerow([global_config.args.model, data_class, f"{mean_rmse}±{std_rmse}", f"{mean_mae}±{std_mae}"])

    logging.info(f"Saved RMSE and MAE metrics to {csv_file_path}")


def set_seeds_and_reproducibility(seed_value):
    """
    Set seeds for reproducibility and configure PyTorch for deterministic behavior.

    Parameters:
    reproducible (bool): Whether to configure the environment for reproducibility.
    seed_value (int): The base seed value to use for RNGs.
    """
    if global_config.args.reproduce:
        # Set seeds with different offsets to avoid correlations
        random.seed(seed_value)
        np.random.seed(seed_value + 1)
        torch.manual_seed(seed_value + 2)
        torch.cuda.manual_seed_all(seed_value + 3)
        # Configure PyTorch for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow PyTorch to optimize for performance
        torch.backends.cudnn.benchmark = True


def convert_to_serializable(val):
    """
    Converts non-serializable objects NumPy data types into serializable native Python types.

    Args:
        val (any): The value to be converted, which can be of any type including
                   NumPy types, lists, or dictionaries.

    Returns:
        any: The converted value in native Python types, maintaining the structure of lists
             and dictionaries.
    """
    if isinstance(val, (np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.float64, np.float32)):
        return float(val)
    elif isinstance(val, list) and len(val) == 1:
        return convert_to_serializable(val[0])
    elif isinstance(val, dict):
        return {k: convert_to_serializable(v) for k, v in val.items()}
    return val


def cartesian_to_polar(coords):
    """
    Converts Cartesian coordinates to polar (2D) or spherical (3D) coordinates based on configuration settings.

    Args:
        coords (list of tuples): A list of tuples representing Cartesian coordinates. Tuples can be either
                                 two-dimensional (x, y) or three-dimensional (x, y, z) based on configuration.

    Returns:
        list of lists: A list containing converted coordinates as [r, theta] or [r, theta, phi] based on dimensionality.
    """
    # Check if the input is a batch of coordinate sets (list of list of lists)
    if all(isinstance(sublist, list) and all(isinstance(item, list) for item in sublist) for sublist in coords):
        # Flatten the batch if it's a list of list of lists
        coords = [item for sublist in coords for item in sublist]

    polar_coords = []
    # print("coords: ", coords)
    # quit()
    if global_config.args.three_dim:
        # if global_config.args.inference:
        #     # print("coords[0][0]: ", coords[0])
        #     for x, y, z in coords[0]:
        #         r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        #         theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
        #         phi = np.arctan2(y, x)
        #         polar_coords.append([r, theta, phi])
        # else:
        for x, y, z in coords:
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)
            phi = np.arctan2(y, x)
            polar_coords.append([r, theta, phi])
    else:
        for x, y in coords:
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            polar_coords.append([r, theta])
    return polar_coords

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    best_model_state,
    train_details,
    val_details
):
    """Save complete training state for recovery."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "best_val_loss": best_val_loss,
        "best_model_state_dict": best_model_state,

        "train_details": train_details,
        "val_details": val_details,

        # Random states for reproducibility
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )
    }

    # Write to a temporary file first to avoid corrupting
    # the previous checkpoint if saving gets interrupted.
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Restore training state from a checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore random states
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())


    if checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )

    return (
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint["best_model_state_dict"],
        checkpoint["train_details"],
        checkpoint["val_details"]
    )
