import torch
from tqdm import tqdm

from new_model_architecture.model import GNN
from new_model_architecture.utils import AverageMeter


# Can be mo
def train_one_epoch(
        model: GNN, 
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device
    ) -> float | None:
    loss_meter: AverageMeter = AverageMeter()
    num_steps = len(train_loader)
    progress_bar = tqdm(train_loader, total=num_steps, desc="Training", leave=False)

    model.train()

    for num_batches, data in enumerate(progress_bar):
        batch = data.to(device)
        
        optimizer.zero_grad()
        gnn_prediction, final_prediction, weight = model(batch)

        loss_gnn = criterion(gnn_prediction, batch.y_pos)
        loss_final = criterion(final_prediction, batch.y_pos)

        loss = (loss_gnn + loss_final) / 2
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.detach(), batch.num_graphs)

    return loss_meter.avg


def validate_one_epoch(
        model: GNN, 
        val_loader,
        criterion,
        device
    ) -> float:
    loss_meter: AverageMeter = AverageMeter()
    num_steps = len(val_loader)
    progress_bar = tqdm(val_loader, total=num_steps, desc="Validation", leave=False)

    model.eval()

    with torch.no_grad():
        for num_batches, data in enumerate(progress_bar):
            batch = data.to(device)
            
            gnn_prediction, final_prediction, weight = model(batch)

            loss_gnn = criterion(gnn_prediction, batch.y_pos)
            loss_final = criterion(final_prediction, batch.y_pos)

            loss = (loss_gnn + loss_final) / 2

            loss_meter.update(loss.item(), batch.num_graphs)

    return loss_meter.avg
