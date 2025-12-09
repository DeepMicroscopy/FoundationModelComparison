# Small example script that runs calc_metrics.py via `uv run` to aggregate and
# compute evaluation metrics for the ViT_S training experiment. It reads results from
# example/train/ViT_S, calculates metrics for a single cross-validation fold
# (num_folds=1), and stores/prints summary metrics under the experiment name
# ViT_S.

uv run calc_metrics.py --model_result_dirs ViT_S --exp_name ViT_S --path_to_results example/train --num_folds 1