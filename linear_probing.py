import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from UNI.uni.downstream.eval_patch_features.metrics import print_metrics
from UNI.uni.downstream.eval_patch_features.linear_probe import eval_linear_probe


SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.2
TRAIN_SIZES = [0.001, 0.01, 0.1, 1.0]

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate extracted patch features with a linear probing across multiple train sizes."
    )
    parser.add_argument(
        "--path_to_features",
        type=Path,
        required=True,
        help="Path to extracted features (e.g. /data/features/ViT_S.pkl).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to identify and save results.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory to save results using model_name (e.g. results/linear_probing/).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help=f"Fraction of data used for testing (default: {TEST_SIZE}).",
    )
    parser.add_argument(
        "--train_sizes",
        nargs="+",
        type=float,
        default=TRAIN_SIZES,
        help=f"Fractions of remaining data used for training (default: {TRAIN_SIZES}).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help=f"Random seeds for resampling (default: {SEEDS}).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    model_name: str = args.model_name
    save_dir: Path = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    test_size: float = args.test_size
    train_sizes: List[float] = args.train_sizes
    seeds: List[int] = args.seeds

    path_to_features: Path = args.path_to_features
    if not path_to_features.is_file():
        raise FileNotFoundError(f"Cannot find {path_to_features}. The file does not exist.")

    with path_to_features.open("rb") as file:
        features = pickle.load(file)

    if "labels" not in features or "embeddings" not in features:
        raise KeyError("Features file must contain 'labels' and 'embeddings' keys.")

    labels = torch.as_tensor(features["labels"], dtype=torch.long)
    embeddings = torch.as_tensor(features["embeddings"])

    num_samples = len(labels)
    if num_samples == 0:
        raise ValueError("No samples found in features.")

    test_size_abs = int(np.floor(num_samples * test_size))
    if test_size_abs <= 0 or test_size_abs >= num_samples:
        raise ValueError(
            f"Invalid test_size={test_size}. Computed test set size {test_size_abs} "
            f"must be between 1 and {num_samples - 1}."
        )

    results: Dict[str, Dict[float, List[Dict[str, Any]]]] = {model_name: {}}

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Sampling test indices 
        test_indices = rng.choice(num_samples, size=test_size_abs, replace=False)
        all_indices = np.arange(num_samples)
        train_pool_indices = np.setdiff1d(all_indices, test_indices)

        for train_size in train_sizes:
            if not (0 < train_size <= 1.0):
                raise ValueError(f"train_size must be in (0, 1], got {train_size}")

            abs_train_size = int(np.floor(len(train_pool_indices) * train_size))
            if abs_train_size <= 0:
                logger.warning(
                    "Skipping train_size=%.4f because it yields 0 training samples.", train_size
                )
                continue

            # Sample training indices
            train_indices = rng.choice(train_pool_indices, size=abs_train_size, replace=False)

            target_train_size = int(0.8 * len(train_indices))
            if target_train_size > 0:
                train_indices = rng.choice(train_indices, size=target_train_size, replace=False)

            train_features = embeddings[train_indices]
            train_labels = labels[train_indices]
            test_features = embeddings[test_indices]
            test_labels = labels[test_indices]

            linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
                train_feats=train_features,
                train_labels=train_labels,
                valid_feats=None,
                valid_labels=None,
                test_feats=test_features,
                test_labels=test_labels,
                max_iter=1000,
                verbose=False,
            )

            results[model_name].setdefault(train_size, []).append(
                {
                    "seed": seed,
                    "metrics": linprobe_eval_metrics,
                    "predictions": linprobe_dump["preds_all"],
                    "targets": linprobe_dump["targets_all"],
                }
            )

            logger.info("_____ seed=%d train_size=%.4f model=%s _____", seed, train_size, model_name)
            logger.info("---- Linear probing -----")
            
            for k, v in linprobe_eval_metrics.items():
                if "report" in k:
                    continue
                logger.info(f"Test {k}: {v:.3f}")

            logger.info("Class counts in training set: %s", np.unique(train_labels, return_counts=True))

    save_file = save_dir / f"{model_name}.pkl"
    with save_file.open("wb") as file:
        pickle.dump(results, file)
    logger.info("Saved results to %s", save_file)


if __name__ == "__main__":
    main(get_args())