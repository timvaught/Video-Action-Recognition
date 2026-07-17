"""Training and evaluation loops with checkpointing and optional W&B logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    grad_clip: float = 1.0,
    scaler: Any | None = None,
    amp: bool = True,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, float]:
    training = optimizer is not None
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    amp_enabled = amp and device.type == "cuda"
    number_of_batches = len(loader)

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (videos, labels, lengths) in enumerate(
            tqdm(loader, leave=False)
        ):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(videos, lengths)
                loss = criterion(logits, labels)

            if training:
                group_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps,
                    number_of_batches - group_start,
                )
                loss_for_backward = loss / group_size
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == number_of_batches
                )

                if scaler is not None and amp_enabled:
                    scaler.scale(loss_for_backward).backward()
                    if should_step:
                        scaler.unscale_(optimizer)
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if should_step:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            batch_size = labels.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("DataLoader produced no examples")
    return total_loss / total_examples, total_correct / total_examples


def train(
    dataloaders: dict[str, DataLoader],
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau,
    device: torch.device,
    checkpoint_dir: str,
    epochs: int = 30,
    wandb_run=None,
    unfreeze_epoch: int = 3,
    early_stopping_patience: int = 6,
    minimum_epochs: int = 1,
    amp: bool = True,
    checkpoint_metadata: dict[str, Any] | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, list[float]]]:
    """Train, validate, checkpoint, and log metrics without touching test data."""
    history_loss = {"train": [], "val": []}
    history_accuracy = {"train": [], "val": []}
    best_weights = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    best_accuracy = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = Path(checkpoint_dir) / "best_model_wts.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    amp_enabled = amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # Compatibility with older supported PyTorch.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(epochs):
        displayed_epoch = epoch + 1
        if (
            unfreeze_epoch > 0
            and displayed_epoch == unfreeze_epoch
            and hasattr(model, "set_backbone_trainable")
        ):
            model.set_backbone_trainable(True)
            print(f"Unfroze video backbone at epoch {displayed_epoch}.")

        train_loss, train_accuracy = _run_epoch(
            model,
            dataloaders["train"],
            criterion,
            device,
            optimizer,
            scaler=scaler,
            amp=amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        validation_loss, validation_accuracy = _run_epoch(
            model,
            dataloaders["val"],
            criterion,
            device,
            amp=amp,
        )

        history_loss["train"].append(train_loss)
        history_loss["val"].append(validation_loss)
        history_accuracy["train"].append(train_accuracy)
        history_accuracy["val"].append(validation_accuracy)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(validation_loss)
        else:
            scheduler.step()

        improved = validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_loss
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            epochs_without_improvement = 0
            best_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            checkpoint = {
                "model_state_dict": best_weights,
                "epoch": epoch,
                "val_accuracy": best_accuracy,
                "val_loss": best_loss,
            }
            if checkpoint_metadata:
                checkpoint.update(checkpoint_metadata)
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1

        learning_rates: dict[str, float] = {}
        for index, group in enumerate(optimizer.param_groups):
            role = str(group.get("role", group.get("name", f"group_{index}")))
            group_learning_rate = float(group["lr"])
            learning_rates[role] = max(
                learning_rates.get(role, float("-inf")),
                group_learning_rate,
            )
        metrics = {
            "epoch": displayed_epoch,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "val/loss": validation_loss,
            "val/accuracy": validation_accuracy,
            "learning_rate/backbone": learning_rates.get("backbone"),
            "learning_rate/head": learning_rates.get("head"),
        }
        print(
            f"Epoch {displayed_epoch:03d}: "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.2%}, "
            f"val loss={validation_loss:.4f}, val acc={validation_accuracy:.2%}, "
            f"head lr={learning_rates.get('head', float('nan')):.2e}, "
            f"backbone lr={learning_rates.get('backbone', float('nan')):.2e}"
        )
        if wandb_run is not None:
            wandb_run.log(metrics)

        stopping_allowed = displayed_epoch >= max(minimum_epochs, unfreeze_epoch, 1)
        if (
            early_stopping_patience > 0
            and stopping_allowed
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping after {displayed_epoch} epochs; "
                f"best validation accuracy={best_accuracy:.2%}."
            )
            break

    model.load_state_dict(best_weights)
    return model, history_loss, history_accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    horizontal_flip_tta: bool = False,
    amp: bool = True,
) -> tuple[float, float, Tensor, Tensor]:
    """Evaluate with temporal multi-clip averaging and optional flip TTA."""
    model.eval()
    total_loss = 0.0
    targets = []
    predictions = []
    amp_enabled = amp and device.type == "cuda"

    with torch.no_grad():
        for videos, labels, lengths in tqdm(loader, leave=False):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(videos, lengths)
                if horizontal_flip_tta:
                    flipped_logits = model(torch.flip(videos, dims=[-1]), lengths)
                    logits = 0.5 * (logits + flipped_logits)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.shape[0]
            targets.append(labels.cpu())
            predictions.append(logits.argmax(dim=1).cpu())

    if not targets:
        raise RuntimeError("Evaluation DataLoader produced no examples")
    target_tensor = torch.cat(targets)
    prediction_tensor = torch.cat(predictions)
    accuracy = (target_tensor == prediction_tensor).float().mean().item()
    return total_loss / len(loader.dataset), accuracy, target_tensor, prediction_tensor
