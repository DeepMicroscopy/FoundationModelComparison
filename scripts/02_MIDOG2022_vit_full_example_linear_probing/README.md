### Full Example: ViT-S Feature Extraction and Linear Probing on MIDOG2022

This full example reproduces one of the experiments from the paper by running the complete feature extraction and linear probing pipeline on the full MIDOG2022 training dataset, using **multiple training sizes** and **multiple random seeds**.

> Note: This experiment can take a **considerable amount of time** to run, depending on your hardware, since it processes a larger dataset and evaluates several configurations.

#### Script Overview

The script performs two main steps:

1. **Feature extraction** with a ViT-S model on the MIDOG2022 training set.
2. **Linear probing** on the extracted features for multiple training set fractions and seeds.

```bash
# Full example: ViT-S feature extraction + linear probing on MIDOG2022 subset. 

echo "=== Step 1/2: Extracting ViT_S features on MIDOG2022 ==="
uv run ../../extract_features.py \
  --path_to_csv_file ../../databases/MIDOG2022_training_tiff.csv \
  --image_dir ../../MIDOG2022/images \
  --out_path full_example \
  --batch_size 32 \
  --num_workers 4 \
  --model ViT_S

echo "=== Step 2/2: Running linear probing on extracted ViT_S features ==="
uv run ../../linear_probing.py \
  --path_to_features full_example/ViT_S_features.pkl \
  --model_name ViT_S \
  --save_dir full_example \
  --train_sizes 0.001 0.01 0.1 1 \
  --seeds 42 43 44 45 46

echo "Done. Results saved to full_example/ViT_S.pkl"
```

#### Prerequisites

- `uv` installed and configured
- Download the MIDOG2022 .tiff dataset from the google-drive as described [here](../../README.md).

#### What the Example Does

1. **Feature Extraction**

    Runs:

        ```bash
        uv run ../../extract_features.py \
        --path_to_csv_file ../../databases/MIDOG2022_training_tiff.csv \
        --image_dir ../../MIDOG2022/images \
        --out_path full_example \
        --batch_size 32 \
        --num_workers 4 \
        --model ViT_S
        ```

    This step:

    - Uses the ViT-S model (`--model ViT_S`)
    - Reads image paths and labels from MIDOG2022_training_tiff.csv
    - Loads images from MIDOG2022/images
    - Extracts features for the full (or large) MIDOG2022 training subset
    - Saves the resulting feature file as: 
        - `full_example/ViT_S_features.pkl`

2. **Linear Probing**

    Runs:

    ```bash
    uv run ../../linear_probing.py \
    --path_to_features full_example/ViT_S_features.pkl \
    --model_name ViT_S \
    --save_dir full_example \
    --train_sizes 0.001 0.01 0.1 1 \
    --seeds 42 43 44 45 46
    ```

    This step:
    - Loads precomputed ViT-S features from `full_example/ViT_S_features.pkl`
    - Trains linear classifiers at multiple relative training set sizes:
        - 0.001 (0.1% of the data)
        - 0.01 (1% of the data)
        - 0.1 (10% of the data)
        - 1 (100% of the data)
    - Uses mulitple random seeds for monte carlo cross validation
    - Saves the aggregated results to:
        - `full_example/ViT_S.pkl`

3. **Usage**

    1. Make the script executable:
    ```bash 
    chmod +x run_midog_vit_full_example.sh
    ```
    2. Run the end-to-end example:
    ```bash 
    ./run_midog_vit_full_example.sh
    ```