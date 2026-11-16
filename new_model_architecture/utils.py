import torch
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


def checkpoint_model(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to save.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        checkpoint_path (str): Path to save the checkpoint file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)