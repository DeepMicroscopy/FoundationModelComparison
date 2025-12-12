### Full Example: ViT-S LoRA Fine-tuning in a Cross-Domain Setting on MIDOGpp + Metric Calculation

This example reproduces a full **cross-domain** experiment from the paper, however this is performed on the extended MIDOGpp dataset. It consists of:

1. **LoRA fine-tuning** of a ViT-S classifier using a tumor-specific / cross-domain training setup.
2. **Metric aggregation** across cross-validation folds to produce a single summarized results file.

> Note: This experiment can take a **long time** to run because it fine-tunes multiple models (multiple random seeds, long training schedule) and then aggregates results across folds.

#### Script Overview

```bash
# Full example: ViT-S fine-tuning with LoRA in cross-domain setting + calculation of metrics 
    
echo "=== Step 1/2: LoRA fine-tuning of ViT-S on MIDOGpp ==="
uv run ../../train_classifier_tumor_specific.py \
    --path_to_csv_file ../../databases/MIDOGpp.csv \
    --image_dir ../../MIDOGpp \
    --checkpoint_path experiment \
    --exp_code ViT_S_LoRA_cross \
    --model_name ViT_S \
    --scheduler \
    --patience 10 \
    --batch_size 16 \
    --augmentation \
    --seeds 42 43 44 45 46 \
    --num_epochs 100 \
    --lora 

echo "=== Step 2/2: Calculation of metrics ==="
uv run ../../calc_metrics.py \
    --model_result_dirs ViT_S_LoRA_cross \
    --exp_name ViT_S_LoRA_Cross \
    --path_to_results experiment \
    --num_folds 5
```

---

#### Prerequisites

- `uv` installed and configured
- Download the MIDOGpp dataset as described [here](../../README.md)

---

#### What the Example Does

##### Step 1: Cross-domain LoRA Fine-tuning

Runs:

```bash
uv run ../../train_classifier_tumor_specific.py \
    --path_to_csv_file ../../databases/MIDOGpp.csv \
    --image_dir ../../MIDOGpp \
    --checkpoint_path experiment \
    --exp_code ViT_S_LoRA_cross \
    --model_name ViT_S \
    --scheduler \
    --patience 10 \
    --batch_size 16 \
    --augmentation \
    --seeds 42 43 44 45 46 \
    --num_epochs 100 \
    --lora
```

This step:

- Fine-tunes a **ViT-S** model (`--model_name ViT_S`) using the **tumor-specific / cross-domain** training procedure implemented in `train_classifier_tumor_specific.py`
- Uses **LoRA** adapters for parameter-efficient fine-tuning (`--lora`)
- Enables data augmentation (`--augmentation`)
- Trains with a scheduler (`--scheduler`) and early-stopping patience of 10 (`--patience 10`)
- Trains for up to 100 epochs (`--num_epochs 100`)
- Repeats training for multiple random seeds (`--seeds 42 43 44 45 46`)

You should expect the script to create an experiment directory structure under `experiment/ViT_S_LoRA_cross/` containing model checkpoints and intermediate results for each tumor and model run combination.

##### Step 2: Metric Calculation / Aggregation

Command:

```bash
uv run ../../calc_metrics.py \
    --model_result_dirs ViT_S_LoRA_cross \
    --exp_name ViT_S_LoRA_cross \
    --path_to_results experiment \
    --num_folds 5
```

This step:

- Collects the outputs produced by the fine-tuning runs under `experiment/`
- Computes results across **5 folds** (`--num_folds 5`)
- Writes an aggregated results file under to `experiment/ViT_S_LoRA_cross/results.pkl`

---


#### Usage

1. Make it executable:

   ```bash
   chmod +x run_midogpp_vit_lora_cross_domain.sh
   ```

2. Run the experiment:

   ```bash
   ./run_midogpp_vit_lora_cross_domain.sh
   ```

This example is intended to reproduce the **cross-domain LoRA fine-tuning** setting and generate a single aggregated metrics file for reporting and comparison.
