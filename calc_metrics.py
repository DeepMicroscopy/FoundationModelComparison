"""
Evaluation metrics computation script for model predictions.

This script computes evaluation metrics based on prediction results stored under
specified model results directories. It iterates through each provided directory,
processes prediction files, computes metrics using predefined evaluation functions,
and saves the aggregated results as pickle files for each directory.

Example:
    python compute_metrics.py \
        --model_results_dirs dir1 dir2 \
        --exp_names exp1 exp2 \
        --path_to_results /path/to/results \
        --num_folds 5
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from UNI.uni.downstream.eval_patch_features import get_eval_metrics

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
        description=(
            "Process experiment results and compute metrics based on stored predictions."
        )
    )

    parser.add_argument(
        "--model_results_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of model results directories (relative to path_to_results).",
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment names (must match the number of model_results_dirs).",
    )
    parser.add_argument(
        "--path_to_results",
        type=str,
        required=True,
        help="Root path to the results directory.",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds to process (default: 5).",
    )
    parser.add_argument(
        "--metric_prefix",
        type=str,
        default="lin_",
        help="Prefix for metric names in the output (default: 'lin_').",
    )

    return parser.parse_args()


def validate_arguments(model_results_dirs: List[str], exp_names: List[str]) -> None:
    """Validate that the number of directories matches the number of experiment names.

    Args:
        model_results_dirs (List[str]): List of model results directories.
        exp_names (List[str]): List of experiment names.

    Raises:
        ValueError: If the lengths of the two lists do not match.
    """
    if len(model_results_dirs) != len(exp_names):
        logger.error(
            "Mismatch: %d model_results_dirs but %d exp_names provided.",
            len(model_results_dirs),
            len(exp_names),
        )
        raise ValueError(
            "The number of model_results_dirs must match the number of exp_names."
        )


def load_fold_results(
    results_file: Path,
) -> pd.DataFrame:
    """Load prediction results from a CSV file for a single fold.

    Args:
        results_file (Path): Path to the CSV file containing predictions.

    Returns:
        pd.DataFrame: DataFrame with columns 'label', 'predicted', and 'probs'.

    Raises:
        FileNotFoundError: If the results file does not exist.
        KeyError: If required columns are missing from the CSV.
    """
    if not results_file.exists():
        logger.error("Results file not found: %s", results_file)
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)

    required_columns = {"label", "predicted", "probs"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error("Missing columns in %s: %s", results_file, missing)
        raise KeyError(f"Missing required columns: {missing}")

    return df


def compute_metrics_for_fold(
    fold_df: pd.DataFrame,
    metric_prefix: str = "lin_",
) -> Dict[str, Any]:
    """Compute evaluation metrics for a single fold.

    Args:
        fold_df (pd.DataFrame): DataFrame containing 'label', 'predicted', and 'probs'.
        metric_prefix (str, optional): Prefix for metric names. Defaults to 'lin_'.

    Returns:
        Dict[str, Any]: Dictionary of computed metrics.
    """
    eval_metrics = get_eval_metrics(
        targets_all=fold_df["label"].values,
        preds_all=fold_df["predicted"].values,
        probs_all=fold_df["probs"].values,
        get_report=True,
        prefix=metric_prefix,
    )

    return eval_metrics


def process_subdirectory(
    subdir_path: Path,
    num_folds: int,
    metric_prefix: str,
) -> List[Dict[str, Any]]:
    """Process all folds within a subdirectory and compute metrics.

    Args:
        subdir_path (Path): Path to the subdirectory containing fold results.
        num_folds (int): Number of folds to process.
        metric_prefix (str): Prefix for metric names.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, one per fold, each containing
            a 'metrics' key with computed evaluation metrics.
    """
    logger.info("Processing subdirectory: %s", subdir_path.name)
    fold_results = []

    for fold_idx in range(num_folds):
        results_file = subdir_path / f"{fold_idx}_results.csv"

        try:
            fold_df = load_fold_results(results_file)
            metrics = compute_metrics_for_fold(fold_df, metric_prefix)
            fold_results.append({"metrics": metrics})
            logger.debug("Fold %d processed successfully", fold_idx)

        except (FileNotFoundError, KeyError) as e:
            logger.warning("Skipping fold %d in %s: %s", fold_idx, subdir_path.name, e)
            continue

    logger.info(
        "Completed processing %d/%d folds for %s",
        len(fold_results),
        num_folds,
        subdir_path.name,
    )
    return fold_results


def process_results_for_dir(
    model_results_dir: str,
    exp_name: str,
    path_to_results: Path,
    num_folds: int,
    metric_prefix: str,
) -> None:
    """Process all subdirectories within a model results directory and save metrics.

    Args:
        model_results_dir (str): Name of the model results directory (relative path).
        exp_name (str): Experiment name for organizing results.
        path_to_results (Path): Root path to the results directory.
        num_folds (int): Number of cross-validation folds to process.
        metric_prefix (str): Prefix for metric names.
    """
    results_dir = path_to_results / model_results_dir

    if not results_dir.exists():
        logger.warning("Directory does not exist: %s. Skipping.", results_dir)
        return

    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not subdirs:
        logger.warning("No subdirectories found in %s. Skipping.", results_dir)
        return

    logger.info(
        "Found %d subdirectories in %s",
        len(subdirs),
        results_dir,
    )

    results = {exp_name: {}}

    for subdir in subdirs:
        subdir_results = process_subdirectory(
            subdir_path=subdir,
            num_folds=num_folds,
            metric_prefix=metric_prefix,
        )
        results[exp_name][subdir.name] = subdir_results

    # Save aggregated results
    output_file = results_dir / "results.pkl"
    logger.info("Saving aggregated results to: %s", output_file)

    with open(output_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Results saved successfully for %s", exp_name)


def main(args: argparse.Namespace) -> None:
    """Main entry point for metrics computation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logger.info("=" * 70)
    logger.info("Evaluation Metrics Computation Pipeline")
    logger.info("=" * 70)

    validate_arguments(args.model_results_dirs, args.exp_names)

    path_to_results = Path(args.path_to_results)

    if not path_to_results.exists():
        logger.error("Root results path does not exist: %s", path_to_results)
        raise NotADirectoryError(f"Root results path not found: {path_to_results}")

    logger.info("Root results path: %s", path_to_results)
    logger.info("Number of experiments to process: %d", len(args.model_results_dirs))
    logger.info("Number of folds per experiment: %d", args.num_folds)

    for model_results_dir, exp_name in zip(args.model_results_dirs, args.exp_names):
        logger.info("-" * 70)
        logger.info("Processing experiment: %s", exp_name)
        logger.info("Model results directory: %s", model_results_dir)

        process_results_for_dir(
            model_results_dir=model_results_dir,
            exp_name=exp_name,
            path_to_results=path_to_results,
            num_folds=args.num_folds,
            metric_prefix=args.metric_prefix,
        )

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