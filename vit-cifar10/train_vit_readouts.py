"""Train three Vision Transformers that differ only in the vector sent to the classifier.

The hyperparameters match notebooks/TrainViT.ipynb. Each run is a fresh model, trained
for 100 epochs, then the next one starts.

  cls     final class token
  mean    mean of the patch tokens
  patch0  first patch token (top-left patch, index 1 after the class token)
  rand    one patch token drawn uniformly from the image, never the class token

Outputs, one directory per readout. --suffix dropout0.2 writes runs/cls_dropout0.2 and the same for mean and patch0.

  runs/<pool>[_suffix]/log.txt
  runs/<pool>[_suffix]/loss.png
  runs/<pool>[_suffix]/accuracy.png
  runs/<pool>[_suffix]/vit_<pool>[_suffix].pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.vit import VisionTransformer

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parents[1] / "data"
RUNS = ROOT / "runs"

IMAGE_SIZE = 32
BATCH_SIZE = 256
PATCH_SIZE = 8
IN_CHANNELS = 3
EMB_SIZE = 384
NUM_ATTENTION_HEADS = 6
NUM_TRANSFORMER_BLOCKS = 8
DROPOUT = 0.1
N_CLASSES = 10
LR = 3e-4
NUM_EPOCHS = 100
VAL_SIZE = 5_000
SEED = 42

POOLS = ("cls", "mean", "patch0")
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_loaders():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_full = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=False,
        transform=train_transform,
    )
    val_full = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=False,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(train_full), generator=generator)
    val_idx = permutation[:VAL_SIZE].tolist()
    train_idx = permutation[VAL_SIZE:].tolist()

    train_loader = DataLoader(
        Subset(train_full, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(
    pool: str,
    dropout: float,
    pos_embed: str,
    patch_size: int,
    emb_size: int,
    num_heads: int,
    device: torch.device,
) -> VisionTransformer:
    return VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=patch_size,
        in_channels=IN_CHANNELS,
        emb_size=emb_size,
        num_attention_heads=num_heads,
        dropout=dropout,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        n_classes=N_CLASSES,
        pool=pool,
        pos_embed=pos_embed,
    ).to(device)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    labels_all = []
    predictions_all = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        labels_all.append(labels)
        predictions_all.append(logits.argmax(dim=1).cpu())
    return torch.cat(labels_all).numpy(), torch.cat(predictions_all).numpy()


def save_plots(history, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [acc * 100 for acc in history["train_acc"]], label="train")
    ax.plot(epochs, [acc * 100 for acc in history["val_acc"]], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)


def log_line(handle, text: str):
    print(text, flush=True)
    handle.write(text + "\n")
    handle.flush()


def train_one(
    pool: str,
    train_loader,
    val_loader,
    device: torch.device,
    dropout: float,
    pos_embed: str,
    patch_size: int,
    emb_size: int,
    num_heads: int,
    run_name: str,
    lr_step_epochs: int = 0,
    lr_gamma: float = 0.1,
):
    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model = build_model(pool, dropout, pos_embed, patch_size, emb_size, num_heads, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = None
    if lr_step_epochs > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_epochs, gamma=lr_gamma
        )
    n_params = sum(p.numel() for p in model.parameters())

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    with log_path.open("w") as log:
        log_line(log, f"pool: {pool}")
        log_line(log, f"parameters: {n_params}")
        log_line(
            log,
            "hyperparams: "
            f"epochs={NUM_EPOCHS} batch_size={BATCH_SIZE} lr={LR} "
            f"image_size={IMAGE_SIZE} patch_size={patch_size} emb_size={emb_size} "
            f"heads={num_heads} blocks={NUM_TRANSFORMER_BLOCKS} "
            f"dropout={dropout} pos_embed={pos_embed} seed={SEED}",
        )
        if scheduler is None:
            log_line(log, "lr schedule: constant")
        else:
            log_line(
                log,
                f"lr schedule: StepLR every {lr_step_epochs} epochs, "
                f"multiply by {lr_gamma}",
            )

        for epoch in range(NUM_EPOCHS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            log_line(
                log,
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Time: {epoch_seconds:.1f}s",
            )
            if scheduler is not None:
                before = optimizer.param_groups[0]["lr"]
                scheduler.step()
                after = optimizer.param_groups[0]["lr"]
                if after != before:
                    log_line(log, f"Learning rate: {after:.6e}")

        labels, predictions = collect_predictions(model, val_loader, device)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
        log_line(log, "")
        log_line(log, f"Validation accuracy: {accuracy:.4f}")
        log_line(log, "Validation classification report:")
        log.write(report)
        if not report.endswith("\n"):
            log.write("\n")
        log.flush()
        print(f"Validation accuracy: {accuracy:.4f}")
        print(report)

    save_plots(history, out_dir)
    checkpoint_path = out_dir / f"vit_{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "image_size": IMAGE_SIZE,
                "patch_size": patch_size,
                "in_channels": IN_CHANNELS,
                "emb_size": emb_size,
                "num_attention_heads": num_heads,
                "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
                "dropout": dropout,
                "n_classes": N_CLASSES,
                "pool": pool,
                "pos_embed": pos_embed,
                "lr": LR,
                "lr_step_epochs": lr_step_epochs,
                "lr_gamma": lr_gamma,
            },
            "history": history,
        },
        checkpoint_path,
    )
    print(f"Saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--pos-embed", choices=("1d", "2d"), default="1d")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--emb-size", type=int, default=EMB_SIZE)
    parser.add_argument("--heads", type=int, default=NUM_ATTENTION_HEADS)
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to each run directory, e.g. dropout0.2 -> runs/cls_dropout0.2",
    )
    parser.add_argument(
        "--lr-step-epochs",
        type=int,
        default=0,
        help="Divide the learning rate by --lr-gamma every this many epochs. 0 keeps a constant rate.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="Factor applied by the step schedule. Ignored when --lr-step-epochs is 0.",
    )
    parser.add_argument(
        "--pools",
        default=",".join(POOLS),
        help="Comma-separated readouts to train. Default is cls,mean,patch0. rand draws a patch token on every forward.",
    )
    args = parser.parse_args()
    pools = tuple(part.strip() for part in args.pools.split(",") if part.strip())
    unknown = [pool for pool in pools if pool not in (*POOLS, "rand")]
    if unknown:
        raise SystemExit(f"unknown pool: {', '.join(unknown)}")
    suffix = f"_{args.suffix}" if args.suffix else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if IMAGE_SIZE % args.patch_size != 0:
        raise SystemExit(f"patch size {args.patch_size} does not divide image size {IMAGE_SIZE}")
    if args.emb_size % args.heads != 0:
        raise SystemExit(f"emb size {args.emb_size} is not divisible by {args.heads} heads")
    print(
        "Training on:", device,
        "dropout:", args.dropout,
        "pos_embed:", args.pos_embed,
        "patch_size:", args.patch_size,
        "emb_size:", args.emb_size,
        "heads:", args.heads,
        "lr_step_epochs:", args.lr_step_epochs,
        "lr_gamma:", args.lr_gamma,
    )
    for pool in pools:
        run_name = f"{pool}{suffix}"
        print(f"\n=== {run_name} ===")
        train_loader, val_loader = build_loaders()
        train_one(
            pool,
            train_loader,
            val_loader,
            device,
            args.dropout,
            args.pos_embed,
            args.patch_size,
            args.emb_size,
            args.heads,
            run_name,
            args.lr_step_epochs,
            args.lr_gamma,
        )


if __name__ == "__main__":
    main()
