### Full Example: ViT-S Fine-tuning with LoRA + Metric Calculation on MIDOGpp

This example reproduces a full LoRA fine-tuning experiment consisting of:

1. **LoRA fine-tuning** of a ViT-S classifier on the MIDOGpp dataset across multiple training sizes and random seeds.
2. **Post-hoc metric aggregation** over cross-validation folds, producing a single results file.

> Note: This experiment can take a **long time** to run because it trains multiple models (several train sizes × several seeds) and then aggregates metrics across folds.

#### Script Overview

```bash
# Full example: ViT-S fine-tuning with LoRA + calculation of metrics 
    
echo "=== Step 1/2: LoRA fine-tuning of ViT-S on MIDOGpp ==="
uv run ../../train_classifier.py \
    --path_to_csv_file ../../databases/MIDOGpp.csv \
    --image_dir ../../MIDOGpp \
    --checkpoint_path experiment \
    --exp_code ViT_S_LoRA \
    --model_name ViT_S \
    --train_sizes 0.001 0.01 0.1 1 \
    --seeds 42 43 44 45 46 \
    --augmentation  \
    --lora \
    --scheduler \

echo "=== Step 2/2: Calculation of metrics ==="
uv run calc_metrics.py \
    --model_result_dirs ViT_S_LoRA \
    --exp_name ViT_S_LoRA \
    --path_to_results experiment \
    --num_folds 5

echo "Done. Results saved to experiment/ViT_S_LoRA/results.pkl"
```

#### Prerequisites

- `uv` installed and configured
- Download the MIDOGpp dataset as described [here](../../README.md)

#### What the Example Does

##### Step 1: LoRA Fine-tuning

Runs:

```bash
uv run ../../train_classifier.py \
    --path_to_csv_file ../../databases/MIDOGpp.csv \
    --image_dir ../../MIDOGpp \
    --checkpoint_path experiment \
    --exp_code ViT_S_LoRA \
    --model_name ViT_S \
    --train_sizes 0.001 0.01 0.1 1 \
    --seeds 42 43 44 45 46 \
    --augmentation \
    --lora \
    --scheduler
```

This step:

- Fine-tunes a **ViT-S** classifier (`--model_name ViT_S`) on **MIDOGpp**
- Uses **LoRA** adapters (`--lora`) for parameter-efficient fine-tuning
- Enables data augmentation (`--augmentation`)
- Uses a learning-rate scheduler (`--scheduler`)
- Runs multiple training configurations:
  - Training set fractions: `0.001`, `0.01`, `0.1`, `1`
  - Random seeds: `42`, `43`, `44`, `45`, `46`
- Writes checkpoints and per-run outputs under the experiment root:
  - `--checkpoint_path experiment`
  - `--exp_code ViT_S_LoRA`

You should expect the script to create an experiment directory structure under `experiment/ViT_S_LoRA/` containing model checkpoints and intermediate results for each fold/seed/train-size combination.

##### Step 2: Metric Calculation / Aggregation

Command:

```bash
uv run calc_metrics.py \
    --model_result_dirs ViT_S_LoRA \
    --exp_name ViT_S_LoRA \
    --path_to_results experiment \
    --num_folds 5
```

This step:

- Collects and aggregates the outputs produced by the training step
- Computes the final metrics across **5 folds** (`--num_folds 5`)
- Saves the aggregated experiment summary to: `experiment/ViT_S_LoRA/results.pkl`

---

#### Outputs

After successful execution, the main artifact is:

- `experiment/ViT_S_LoRA/results.pkl` — aggregated metrics for the LoRA fine-tuning experiment across all specified training sizes, seeds, and folds.

Additional artifacts (checkpoints, logs, per-run predictions/metrics) are typically stored under:

- `experiment/ViT_S_LoRA/` (structure depends on the training script implementation)

---

#### Usage

1. Make it executable:

   ```bash
   chmod +x run_midogpp_vit_lora_full_example.sh
   ```

2. Run the experiment:

   ```bash
   ./run_midogpp_vit_lora_full_example.sh
   ```

This example is intended to reproduce a full fine-tuning experiment with LoRA and produce a single aggregated metrics file suitable for reporting and comparison. 
```