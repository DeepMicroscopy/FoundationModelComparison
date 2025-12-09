"""
Training script for mitotic figure classification models.

This script trains a classifier on histopathology patches with support for:
    - Multiple training data fractions 
    - Multiple random seeds for monte-carlo cross-validation
    - Early stopping and learning rate scheduling
    - LoRA fine-tuning
    - Data augmentation
    - TensorBoard logging

Example:
    python train_classifier.py \
        --path_to_csv_file /path/to/data.csv \
        --image_dir /path/to/images \
        --checkpoint_path /path/to/checkpoints \
        --exp_code experiment_name \
        --model_name resnet50 \
        --train_sizes 0.01,0.1,1.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from src.dataset import Mitosis_Base_Dataset
from src.classifier import Classifier
from src.utils import collate_fn

# Default hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 4
PATCH_SIZE = 224
TEST_PORTION = 0.2
PSEUDO_EPOCH_LENGTH = 1280
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 20
TRAIN_SIZES = [0.001, 0.01, 0.1, 1.0]
SEEDS = [42, 43, 44, 45, 46]

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a classifier on histopathology patches."
    )

    # Required arguments
    parser.add_argument(
        "--path_to_csv_file",
        type=str,
        required=True,
        help="Path to CSV file containing dataset information.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory containing the images.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Directory where checkpoints and results will be saved.",
    )

    # Experiment configuration
    parser.add_argument(
        "--exp_code",
        type=str,
        default="default_experiment",
        help="Experiment code/name for organizing results.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="Name of the model architecture to use.",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--train_sizes",
        nargs="+",
        type=float,
        default=TRAIN_SIZES,
        help=f"Comma-separated list of training data fractions (e.g., {TRAIN_SIZES}).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help=f"Seeds to indicate how many repititions per configuration (e.g. {SEEDS})."
    )

    # Data configuration
    parser.add_argument(
        "--patch_size",
        type=int,
        default=PATCH_SIZE,
        help=f"Patch size in pixels (default: {PATCH_SIZE}).",
    )
    parser.add_argument(
        "--test_portion",
        type=float,
        default=TEST_PORTION,
        help=f"Fraction of data to use for testing (default: {TEST_PORTION}).",
    )
    parser.add_argument(
        "--pseudo_epoch_length",
        type=int,
        default=PSEUDO_EPOCH_LENGTH,
        help=f"Number of samples per pseudo-epoch (default: {PSEUDO_EPOCH_LENGTH}).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of DataLoader workers (default: {NUM_WORKERS}).",
    )

    # Regularization and optimization
    parser.add_argument(
        "--augmentation",
        action="store_true",
        default=False,
        help="Enable data augmentation during training.",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        default=False,
        help="Use OneCycleLR learning rate scheduler.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="Enable early stopping based on validation loss.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Early stopping patience (default: {PATIENCE}).",
    )
    parser.add_argument(
        "--gradient_clipping",
        action="store_true",
        default=False,
        help="Enable gradient clipping (max norm 0.1).",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        help="Use LoRA (Low-Rank Adaptation) for fine-tuning.",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: 'cuda').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode with reduced dataset size.",
    )

    return parser.parse_args()


def save_args_to_yaml(args: argparse.Namespace, output_path: Path) -> None:
    """Save command-line arguments to a YAML file.

    Args:
        args (argparse.Namespace): Parsed arguments.
        output_path (Path): Path to the output YAML file.
    """
    logger.info("Saving arguments to: %s", output_path)
    with open(output_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)


def print_model_parameters(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Print detailed information about model parameters.

    Args:
        model (torch.nn.Module): PyTorch model.
        model_name (str, optional): Name to display in output. Defaults to "Model".
    """
    trainable_params = 0
    non_trainable_params = 0
    total_params = 0

    logger.info("=" * 70)
    logger.info("%s Parameter Summary", model_name)
    logger.info("=" * 70)

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
    non_trainable_pct = (
        (non_trainable_params / total_params * 100) if total_params > 0 else 0
    )

    logger.info("Trainable params:     %12s (%6.2f%%)", f"{trainable_params:,}", trainable_pct)
    logger.info("Non-trainable params: %12s (%6.2f%%)", f"{non_trainable_params:,}", non_trainable_pct)
    logger.info("Total params:         %12s", f"{total_params:,}")

    # Memory estimation
    param_size_mb = total_params * 4 / (1024**2)  # float32
    logger.info("Estimated size (FP32): %.2f MB", param_size_mb)
    logger.info("Estimated size (FP16): %.2f MB", param_size_mb / 2)
    logger.info("=" * 70)


def split_dataset(
    df: pd.DataFrame,
    train_size: float,
    test_portion: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.

    Args:
        df (pd.DataFrame): Full dataset.
        train_size (float): Fraction of data to use for training (after test split).
        test_portion (float): Fraction of data to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames.
    """
    np.random.seed(seed)

    # Split test set
    test_indices = np.random.choice(
        df.index, size=int(len(df) * test_portion), replace=False
    )
    test_df = df.loc[test_indices]
    remaining_df = df.drop(test_indices)

    # Select training samples based on train_size
    train_indices = np.random.choice(
        remaining_df.index, size=int(len(remaining_df) * train_size), replace=False
    )
    train_df = remaining_df.loc[train_indices]

    # Split validation set from training set
    val_indices = np.random.choice(
        train_df.index, size=int(len(train_df) * test_portion), replace=False
    )
    val_df = train_df.loc[val_indices]
    train_df = train_df.drop(val_indices)

    # Verify no overlaps
    assert len(set(train_df.index) & set(val_df.index)) == 0, "Train/val overlap detected"
    assert len(set(train_df.index) & set(test_df.index)) == 0, "Train/test overlap detected"
    assert len(set(val_df.index) & set(test_df.index)) == 0, "Val/test overlap detected"

    logger.info("Dataset split - Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    return train_df, val_df, test_df


def create_dataloaders(
    df: pd.DataFrame,
    image_dir: Path,
    model: Classifier,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Args:
        df (pd.DataFrame): DataFrame with 'split' column indicating train/val/test.
        image_dir (Path): Root directory containing images.
        model (Classifier): Model instance (for accessing input transforms).
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    base_dataset = Mitosis_Base_Dataset(csv_file=df, image_dir=image_dir)

    # Base transform from model
    base_transform = model.input_transform

    # Training augmentations
    if args.augmentation:
        train_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0))], p=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=180)], p=0.5),
            *base_transform.transforms,
        ])
        logger.info("Data augmentation enabled")
    else:
        train_transform = base_transform
        logger.info("Data augmentation disabled")

    # Create datasets
    train_ds = base_dataset.return_split(
        split="train",
        patch_size=args.patch_size,
        level=0,
        transforms=train_transform,
        pseudo_epoch_length=args.pseudo_epoch_length,
    )
    val_ds = base_dataset.return_split(
        split="val",
        patch_size=args.patch_size,
        level=0,
        transforms=base_transform,
        pseudo_epoch_length=args.pseudo_epoch_length,
    )
    test_ds = base_dataset.return_split(
        split="test",
        patch_size=args.patch_size,
        level=0,
        transforms=base_transform,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_ds.collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_ds.collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    clip_grad: bool = False,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        train_loader (DataLoader): Training DataLoader.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler.
        clip_grad (bool, optional): Whether to clip gradients. Defaults to False.

    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(model.device if hasattr(model, 'device') else 'cuda')
        labels = labels.to(model.device if hasattr(model, 'device') else 'cuda')

        optimizer.zero_grad()
        logits, _, y_hat = model(images)

        # Handle single-sample batches
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
            logits = logits.unsqueeze(0)

        loss = criterion(logits, labels.float())

        if clip_grad:
            clip_grad_norm_(model.parameters(), max_norm=0.1)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += (y_hat == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader: DataLoader,
) -> Tuple[float, float]:
    """Validate the model.

    Args:
        model (torch.nn.Module): Model to validate.
        criterion (torch.nn.Module): Loss function.
        val_loader (DataLoader): Validation DataLoader.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(model.device if hasattr(model, 'device') else 'cuda')
            labels = labels.to(model.device if hasattr(model, 'device') else 'cuda')

            logits, _, y_hat = model(images)

            # Handle single-sample batches
            if y_hat.dim() == 0:
                y_hat = y_hat.unsqueeze(0)
                logits = logits.unsqueeze(0)

            loss = criterion(logits, labels.float())

            running_loss += loss.item()
            total += labels.size(0)
            correct += (y_hat == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
) -> pd.DataFrame:
    """Test the model and return predictions.

    Args:
        model (torch.nn.Module): Model to test.
        test_loader (DataLoader): Test DataLoader.

    Returns:
        pd.DataFrame: DataFrame containing predictions with columns:
            ['file', 'x', 'y', 'label', 'predicted', 'probs'].
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, labels, files, coords in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(model.device if hasattr(model, 'device') else 'cuda')
            labels = labels.to(model.device if hasattr(model, 'device') else 'cuda')

            logits, y_prob, y_hat = model(images)

            # Handle single-sample batches
            if y_prob.dim() == 0:
                y_prob = y_prob.unsqueeze(0)
                y_hat = y_hat.unsqueeze(0)

            for file, coord, label, pred, prob in zip(
                files,
                coords.cpu().numpy(),
                labels.cpu().numpy(),
                y_hat.cpu().numpy(),
                y_prob.cpu().numpy(),
            ):
                results.append({
                    "file": file,
                    "x": coord[0],
                    "y": coord[1],
                    "label": label,
                    "predicted": pred,
                    "probs": prob,
                })

    return pd.DataFrame(results)


def train_single_run(
    df: pd.DataFrame,
    args: argparse.Namespace,
    train_size: float,
    run_idx: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Train a single model run with a specific seed and training size.

    Args:
        df (pd.DataFrame): Full dataset.
        args (argparse.Namespace): Parsed arguments.
        train_size (float): Fraction of training data to use.
        run_idx (int): Run index for logging.
        seed (int): Random seed.
        output_dir (Path): Directory to save results.
    """
    logger.info("=" * 70)
    logger.info("Run %d | Seed: %d | Train size: %.3f", run_idx, seed, train_size)
    logger.info("=" * 70)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Split dataset
    train_df, val_df, test_df = split_dataset(df, train_size, args.test_portion, seed)

    # Assign split labels
    df_copy = df.copy()
    df_copy["split"] = "NONE"
    df_copy.loc[train_df.index, "split"] = "train"
    df_copy.loc[val_df.index, "split"] = "val"
    df_copy.loc[test_df.index, "split"] = "test"

    # Save split
    split_file = output_dir / f"{run_idx}_split.csv"
    df_copy.to_csv(split_file, index=False)
    logger.info("Split saved to: %s", split_file)

    # Debug mode: reduce test set
    if args.debug:
        logger.warning("Debug mode: reducing test set to 7 samples")
        test_df = df_copy[df_copy["split"] == "test"].head(7)
        df_copy = df_copy[df_copy["split"] != "test"]
        df_copy = pd.concat([df_copy, test_df])

    # Initialize model
    logger.info("Initializing model: %s", args.model_name)
    model = Classifier(args.model_name, args.lora)
    model.to(args.device)
    print_model_parameters(model, args.model_name)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df_copy, Path(args.image_dir), model, args
    )

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Initialize scheduler
    scheduler = None
    if args.scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=args.num_epochs,
        )
        logger.info("OneCycleLR scheduler enabled")

    # Initialize TensorBoard
    log_dir = output_dir / str(run_idx)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info("TensorBoard logs: %s", log_dir)

    # Training loop
    best_loss = np.inf
    trigger_times = 0
    best_model_state = None

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, criterion, train_loader, scheduler, args.gradient_clipping
        )
        val_loss, val_acc = validate(model, criterion, val_loader)

        logger.info(
            "Epoch %3d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch + 1,
            args.num_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Early stopping / model saving
        if val_loss < best_loss:
            logger.info("Validation loss improved: %.4f → %.4f", best_loss, val_loss)
            best_loss = val_loss
            trigger_times = 0

            if args.lora:
                model.model.save_pretrained(output_dir / str(run_idx))
            else:
                best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if args.early_stopping:
                logger.info("Early stopping counter: %d/%d", trigger_times, args.patience)

        if args.early_stopping and trigger_times >= args.patience:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

        # Resample training patches
        train_loader.dataset.resample_patches()

    writer.close()

    # Save best model
    if not args.lora and best_model_state is not None:
        model_path = output_dir / f"{run_idx}.pth"
        torch.save(best_model_state, model_path)
        logger.info("Best model saved to: %s", model_path)

    # Load best model for testing
    if best_model_state is not None:
        if args.lora:
            model.load_pretrained_lora_model(args.model_name, output_dir / str(run_idx))
            model.to(args.device)
        else:
            model.load_state_dict(best_model_state)

    # Test
    logger.info("Running test evaluation...")
    results_df = test(model, test_loader)
    results_file = output_dir / f"{run_idx}_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info("Test results saved to: %s", results_file)

    # Log class distribution
    unique, counts = np.unique(train_df["label"], return_counts=True)
    logger.info("Training class distribution: %s", dict(zip(unique, counts)))


def main(args: argparse.Namespace) -> None:
    """Main training loop across multiple train sizes and seeds.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    logger.info("=" * 70)
    logger.info("Training Pipeline")
    logger.info("=" * 70)
    logger.info("Experiment: %s", args.exp_code)
    logger.info("Model: %s", args.model_name)
    logger.info("Device: %s", args.device)
    logger.info("Seeds: %s", args.seeds)
    logger.info("Train sizes: %s", args.train_sizes)

    # Set matmul precision for performance
    torch.set_float32_matmul_precision("medium")

    # Load dataset
    logger.info("Loading dataset from: %s", args.path_to_csv_file)
    df = pd.read_csv(args.path_to_csv_file)
    logger.info("Dataset loaded with %d samples", len(df))

    # Save arguments
    exp_dir = Path(args.checkpoint_path) / args.exp_code
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_args_to_yaml(args, exp_dir / "args.yaml")

    # Loop over training sizes
    for train_size in args.train_sizes:
        logger.info("-" * 70)
        logger.info("Training with %.1f%% of data", train_size * 100)
        logger.info("-" * 70)

        output_dir = exp_dir / str(train_size)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Loop over seeds
        for run_idx, seed in enumerate(args.seeds):
            train_single_run(df, args, train_size, run_idx, seed, output_dir)

    logger.info("=" * 70)
    logger.info("All training runs completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        parsed_args = parse_args()
        main(parsed_args)
    except Exception:
        logger.exception("Training pipeline failed:")
        sys.exit(1)