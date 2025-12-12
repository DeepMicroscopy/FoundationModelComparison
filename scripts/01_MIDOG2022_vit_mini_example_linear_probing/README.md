### Mini Example: ViT-S Feature Extraction and Linear Probing on MIDOG2022

This mini example provides an end-to-end sanity check that the linear probing pipeline runs correctly on a small subset of the MIDOG2022 dataset. It consists of:

1. **Feature extraction** from a ViT-S model.
2. **Linear probing** on the extracted features.

Both steps are executed via a single shell script.

#### Prerequisites

- `uv` installed and configured
- Download the MIDOG2022 .tiff dataset from the google-drive as described [here](../../README.md).

#### What the Example Does

1. **Feature Extraction**

   Runs:

    ```bash
    uv run ../../extract_features.py \
    --path_to_csv_file ../../databases/MIDOG2022_training_debug_tiff.csv \
    --image_dir ../../MIDOG2022/images \
    --out_path mini_example \
    --batch_size 32 \
    --num_workers 4 \
    --model ViT_S
    ```

2. **Linear Probing**

    Runs:
    ```bash
    uv run ../../linear_probing.py \
        --path_to_features example/MIDOG2022/features/ViT_S.pkl \
        --model_name ViT_S \
        --save_dir example/MIDOG2022/results \
        --train_sizes 1 \
        --seeds 43
    ```

    This performs linear probing on the extracted ViT-S features using:
        - 100% of the training data
        - A single random seed 


3. **Usage**

    1. Make the script executable:
    ```bash 
    chmod +x run_midog_vit_mini_example.sh
    ```
    2. Run the end-to-end example:
    ```bash 
    ./run_midog_vit_mini_example.sh
    ```