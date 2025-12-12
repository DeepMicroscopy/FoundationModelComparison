"""
Evaluation metrics computation script.

This script computes evaluation metrics based on prediction results stored under
specified model results directories. It supports two modes:
1. Standard evaluation: Computes metrics for each fold directly
2. Cross-domain evaluation: Computes metrics across different tumor types

Example (standard mode):
    python calc_metrics.py \
        --model_result_dirs dir1 dir2 \
        --exp_names exp1 exp2 \
        --path_to_results /path/to/results \
        --num_folds 5

Example (cross-domain mode):
    python calc_metrics.py \
        --model_result_dirs MIDOG_cross_ViT_S_DINOv3 \
        --exp_names cross_domain_exp \
        --path_to_results results_baseline \
        --num_folds 5 \
        --cross_domain \
        --tumor_info_csv data/MIDOG2022_training_tumor_types.csv \
        --patch_size 224
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.metrics import get_eval_metrics

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
        "--model_result_dirs",
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
        help="List of experiment names (must match the number of model_result_dirs).",
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
    
    # Cross-domain evaluation arguments
    parser.add_argument(
        "--cross_domain",
        action="store_true",
        help="Enable cross-domain evaluation mode.",
    )
    parser.add_argument(
        "--tumor_info_csv",
        type=str,
        help="Path to CSV file containing tumor type information (required for cross-domain mode).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch size for coordinate adjustment in cross-domain mode (default: 224).",
    )
    parser.add_argument(
        "--tumor_types",
        type=str,
        nargs="+",
        help="List of tumor types for cross-domain evaluation. If not provided, will be inferred from subdirectories.",
    )

    return parser.parse_args()


def validate_arguments(
    model_result_dirs: List[str],
    exp_names: List[str],
    cross_domain: bool,
    tumor_info_csv: Optional[str],
) -> None:
    """Validate command-line arguments.

    Args:
        model_result_dirs (List[str]): List of model results directories.
        exp_names (List[str]): List of experiment names.
        cross_domain (bool): Whether cross-domain mode is enabled.
        tumor_info_csv (Optional[str]): Path to tumor info CSV.

    Raises:
        ValueError: If validation fails.
    """
    if len(model_result_dirs) != len(exp_names):
        logger.error(
            "Mismatch: %d model_result_dirs but %d exp_names provided.",
            len(model_result_dirs),
            len(exp_names),
        )
        raise ValueError(
            "The number of model_result_dirs must match the number of exp_names."
        )
    
    if cross_domain and not tumor_info_csv:
        logger.error("Cross-domain mode requires --tumor_info_csv argument.")
        raise ValueError("--tumor_info_csv is required when --cross_domain is enabled.")


def load_tumor_info(tumor_info_csv: str, patch_size: int) -> pd.DataFrame:
    """Load and preprocess tumor type information.

    Args:
        tumor_info_csv (str): Path to the tumor info CSV file.
        patch_size (int): Patch size for coordinate adjustment.

    Returns:
        pd.DataFrame: DataFrame with tumor information and identifiers.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(tumor_info_csv)
    if not csv_path.exists():
        logger.error("Tumor info CSV not found: %s", csv_path)
        raise FileNotFoundError(f"Tumor info CSV not found: {csv_path}")
    
    logger.info("Loading tumor info from: %s", csv_path)
    info = pd.read_csv(csv_path)
    info['identifier'] = info.apply(
        lambda row: f"{row['filename']}_{int(row['x'])}_{int(row['y'])}",
        axis=1
    )
    logger.info("Loaded tumor info with %d entries", len(info))
    return info


def load_fold_results(results_file: Path) -> pd.DataFrame:
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


def process_subdirectory_standard(
    subdir_path: Path,
    num_folds: int,
    metric_prefix: str,
) -> List[Dict[str, Any]]:
    """Process all folds within a subdirectory (standard mode).

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


def process_subdirectory_cross_domain(
    subdir_path: Path,
    subdir_name: str,
    tumor_types: List[str],
    num_folds: int,
    metric_prefix: str,
    tumor_info: pd.DataFrame,
    patch_size: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process all folds within a subdirectory (cross-domain mode).

    Args:
        subdir_path (Path): Path to the subdirectory containing fold results.
        subdir_name (str): Name of the subdirectory (source tumor type).
        tumor_types (List[str]): List of all tumor types.
        num_folds (int): Number of folds to process.
        metric_prefix (str): Prefix for metric names.
        tumor_info (pd.DataFrame): DataFrame with tumor type information.
        patch_size (int): Patch size for coordinate adjustment.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping target tumor types
            to lists of fold results.
    """
    logger.info("Processing subdirectory (cross-domain): %s", subdir_name)
    cross_domain_results = {}

    for target_type in tumor_types:
        cross_domain_results[target_type] = []
        
        for fold_idx in range(num_folds):
            results_file = subdir_path / f"{fold_idx}_results.csv"
            
            try:
                res = pd.read_csv(results_file)
                
                # Create identifier for merging
                res['identifier'] = res.apply(
                    lambda row: f"{row['file']}_{row['x'] + (patch_size//2)}_{row['y'] + (patch_size//2)}",
                    axis=1
                )
                
                # Merge with tumor info and filter by target type
                res = res.merge(tumor_info, on='identifier', how='inner')[
                    ['label_x', 'probs', 'predicted', 'tumortype']
                ]
                res = res[res['tumortype'] == target_type][['label_x', 'probs', 'predicted']]
                res = res.rename(columns={'label_x': 'label'})
                
                if len(res) == 0:
                    logger.warning(
                        "No samples found for %s -> %s fold %d",
                        subdir_name, target_type, fold_idx
                    )
                    continue
                
                # Compute metrics
                eval_res = get_eval_metrics(
                    targets_all=res['label'].values,
                    preds_all=res['predicted'].values,
                    probs_all=res['probs'].values,
                    get_report=True,
                    prefix=metric_prefix,
                )
                
                cross_domain_results[target_type].append({'metrics': eval_res})
                logger.debug(
                    "Fold %d processed: %s -> %s (%d samples)",
                    fold_idx, subdir_name, target_type, len(res)
                )
                
            except (FileNotFoundError, KeyError) as e:
                logger.warning(
                    "Skipping fold %d for %s -> %s: %s",
                    fold_idx, subdir_name, target_type, e
                )
                continue
    
    logger.info("Completed cross-domain processing for %s", subdir_name)
    return cross_domain_results


def process_results_for_dir_standard(
    model_results_dir: str,
    exp_name: str,
    path_to_results: Path,
    num_folds: int,
    metric_prefix: str,
) -> None:
    """Process all subdirectories (standard mode) and save metrics.

    Args:
        model_results_dir (str): Name of the model results directory.
        exp_name (str): Experiment name.
        path_to_results (Path): Root path to the results directory.
        num_folds (int): Number of cross-validation folds.
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

    logger.info("Found %d subdirectories in %s", len(subdirs), results_dir)

    results = {exp_name: {}}

    for subdir in subdirs:
        subdir_results = process_subdirectory_standard(
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


def process_results_for_dir_cross_domain(
    model_results_dir: str,
    exp_name: str,
    path_to_results: Path,
    num_folds: int,
    metric_prefix: str,
    tumor_info: pd.DataFrame,
    patch_size: int,
    tumor_types: Optional[List[str]] = None,
) -> None:
    """Process all subdirectories (cross-domain mode) and save metrics.

    Args:
        model_results_dir (str): Name of the model results directory.
        exp_name (str): Experiment name.
        path_to_results (Path): Root path to the results directory.
        num_folds (int): Number of cross-validation folds.
        metric_prefix (str): Prefix for metric names.
        tumor_info (pd.DataFrame): DataFrame with tumor type information.
        patch_size (int): Patch size for coordinate adjustment.
        tumor_types (Optional[List[str]]): List of tumor types. If None, inferred from subdirectories.
    """
    results_dir = path_to_results / model_results_dir

    if not results_dir.exists():
        logger.warning("Directory does not exist: %s. Skipping.", results_dir)
        return

    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not subdirs:
        logger.warning("No subdirectories found in %s. Skipping.", results_dir)
        return

    # Infer tumor types from subdirectories if not provided
    if tumor_types is None:
        tumor_types = [d.name for d in subdirs]
        logger.info("Inferred tumor types from subdirectories: %s", tumor_types)
    else:
        logger.info("Using provided tumor types: %s", tumor_types)

    logger.info("Found %d subdirectories in %s", len(subdirs), results_dir)

    results = {}

    for subdir in subdirs:
        subdir_name = subdir.name
        if subdir_name not in tumor_types:
            logger.warning("Subdirectory %s not in tumor_types list. Skipping.", subdir_name)
            continue
        
        cross_domain_results = process_subdirectory_cross_domain(
            subdir_path=subdir,
            subdir_name=subdir_name,
            tumor_types=tumor_types,
            num_folds=num_folds,
            metric_prefix=metric_prefix,
            tumor_info=tumor_info,
            patch_size=patch_size,
        )
        results[subdir_name] = cross_domain_results

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
    logger.info("Mode: %s", "Cross-Domain" if args.cross_domain else "Standard")
    logger.info("=" * 70)

    validate_arguments(
        args.model_result_dirs,
        args.exp_names,
        args.cross_domain,
        args.tumor_info_csv,
    )

    path_to_results = Path(args.path_to_results)

    if not path_to_results.exists():
        logger.error("Root results path does not exist: %s", path_to_results)
        raise NotADirectoryError(f"Root results path not found: {path_to_results}")

    logger.info("Root results path: %s", path_to_results)
    logger.info("Number of experiments to process: %d", len(args.model_result_dirs))
    logger.info("Number of folds per experiment: %d", args.num_folds)

    # Load tumor info if in cross-domain mode
    tumor_info = None
    if args.cross_domain:
        tumor_info = load_tumor_info(args.tumor_info_csv, args.patch_size)

    for model_results_dir, exp_name in zip(args.model_result_dirs, args.exp_names):
        logger.info("-" * 70)
        logger.info("Processing experiment: %s", exp_name)
        logger.info("Model results directory: %s", model_results_dir)

        if args.cross_domain:
            process_results_for_dir_cross_domain(
                model_results_dir=model_results_dir,
                exp_name=exp_name,
                path_to_results=path_to_results,
                num_folds=args.num_folds,
                metric_prefix=args.metric_prefix,
                tumor_info=tumor_info,
                patch_size=args.patch_size,
                tumor_types=args.tumor_types,
            )
        else:
            process_results_for_dir_standard(
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