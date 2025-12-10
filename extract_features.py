"""
Feature extraction script for mitotic figure patches.

This script:
    1. Loads a pretrained model and its corresponding inference transforms.
    2. Builds a dataset from a CSV and image directory.
    3. Extracts patch-level features for all entries.
    4. Saves the resulting features dictionary as a pickle file.

Example:
    python extract_features.py \
        --path_to_csv_file /path/to/patches.csv \
        --image_dir /path/to/images \
        --out_path /path/to/output_dir \
        --model virchow2 \
        --device cuda
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from src.dataset import Mitosis_Base_Dataset
from src.utils import (
    collate_fn,
    extract_patch_features_from_dataloader,
    load_model_and_transforms,
    return_forward,
)

# Necessary for some multi-worker dataloading setups on some systems
torch.multiprocessing.set_sharing_strategy("file_system")

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
        description="Extract patch-level features using a pretrained model."
    )

    parser.add_argument(
        "--path_to_csv_file",
        type=str,
        required=True,
        help="Path to CSV file describing patches / images.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory containing the images referenced in the CSV.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Directory where the output pickle file will be saved.",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch size in pixels used by the dataset / model (default: 224).",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help=(
            "Input size in pixels expected by the model (default: 224). "
            "Typically matches patch_size unless the model requires otherwise."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "Swin_base",
            "ViT_H",
            "ViT_S",
            "ViT_S_DINOv3",
            "ViT_tiny",
            "convnext_base",
            "densenet_121",
            "efficientnet_b0", "efficientnet_b3", "efficientnet_b7",
            "gigapath",
            "hoptimus",
            "phikon",
            "resnet50",
            "uni",
            "virchow",
            "virchow2"
        ],
        help="Name of the pretrained model / backbone to use for feature extraction.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the DataLoader (default: 32).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for DataLoader (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device, e.g. 'cuda' or 'cpu' (default: 'cuda').",
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Validate input and output paths.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        NotADirectoryError: If the image directory does not exist.
    """
    csv_path = Path(args.path_to_csv_file)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_path)

    if not csv_path.is_file():
        logger.error("CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    # Create output directory if it does not exist
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", out_dir)


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load CSV into a DataFrame and enforce 'split' column as 'test'.

    Args:
        csv_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with a 'split' column set to 'test'.
    """
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Ensure 'split' column exists and is set to 'test' for all rows
    df["split"] = "test"

    logger.info("CSV loaded with %d entries", len(df))
    return df


def build_dataloader(
    df: pd.DataFrame,
    image_dir: Path,
    transforms: Any,
    patch_size: int,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    """Build the DataLoader for feature extraction.

    Args:
        df (pd.DataFrame): DataFrame describing the dataset.
        image_dir (Path): Root directory containing the images.
        transforms (Any): Model-specific inference-time transform pipeline.
        patch_size (int): Patch size in pixels used by the dataset.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: DataLoader ready for inference / feature extraction.
    """
    logger.info("Initializing dataset...")
    base_dataset = Mitosis_Base_Dataset(
        csv_file=df,
        image_dir=image_dir,
    )

    dataset = base_dataset.return_split(
        split="test",
        patch_size=patch_size,
        level=0,
        transforms=transforms,
    )

    logger.info(
        "Building DataLoader (batch_size=%d, num_workers=%d)...",
        batch_size,
        num_workers,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info("DataLoader initialized with %d samples", len(dataset))
    return dataloader


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    model_name: str,
    device: str,
) -> Dict[str, Any]:
    """Extract patch-level features using the given model and dataloader.

    Args:
        model (torch.nn.Module): Pretrained model backbone.
        dataloader (torch.utils.data.DataLoader): Dataloader providing patches to the model.
        model_name (str): Identifier for the model (used to select the appropriate
            forward function).
        device (str): Device on which the model is running, e.g. 'cuda' or 'cpu'.

    Returns:
        Dict[str, Any]: Dictionary containing extracted features and any associated
        metadata.
    """
    logger.info("Moving model to device: %s", device)
    model.to(device)
    model.eval()

    logger.info("Extracting features...")
    forward_fn = return_forward(model_name)

    with torch.no_grad():
        outputs = extract_patch_features_from_dataloader(
            model=model,
            dataloader=dataloader,
            forward_fn=forward_fn,
        )

    logger.info("Feature extraction completed")
    return outputs


def save_features(features: Dict[str, Any], out_dir: Path, model_name: str) -> Path:
    """Save extracted features to a pickle file.

    Args:
        features (Dict[str, Any]): Extracted features dictionary.
        out_dir (Path): Output directory where the file will be saved.
        model_name (str): Name of the model, used in the filename.

    Returns:
        Path: Full path to the written pickle file.
    """
    out_path = out_dir / f"{model_name}_features.pkl"

    logger.info("Saving features to: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        size_mb = out_path.stat().st_size / 1e6
        logger.info("Features successfully saved (%.2f MB)", size_mb)
    except OSError:
        logger.warning("Features saved, but could not determine file size.")

    return out_path


def main(args: argparse.Namespace) -> None:
    """Main entry point for feature extraction.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logger.info("=" * 70)
    logger.info("Feature Extraction Pipeline")
    logger.info("=" * 70)

    validate_paths(args)

    csv_path = Path(args.path_to_csv_file)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_path)

    logger.info("Device: %s", args.device)
    logger.info("Model: %s", args.model)
    logger.info("Patch size: %d", args.patch_size)
    logger.info("Batch size: %d", args.batch_size)

    # Load model and its inference-time transforms
    logger.info("Loading model: %s", args.model)
    model, transforms = load_model_and_transforms(args.model)
    logger.info("Model '%s' loaded successfully", args.model)

    # Load dataframe
    df = load_dataframe(csv_path)

    # Build dataloader
    dataloader = build_dataloader(
        df=df,
        image_dir=image_dir,
        transforms=transforms,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Feature extraction
    features = extract_features(
        model=model,
        dataloader=dataloader,
        model_name=args.model,
        device=args.device,
    )

    # Save features
    _ = save_features(features, out_dir, args.model)

    logger.info("=" * 70)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        parsed_args = parse_args()
        main(parsed_args)
    except Exception:
        logger.exception("Pipeline failed with error:")
        sys.exit(1)