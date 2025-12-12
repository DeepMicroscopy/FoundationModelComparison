# Experiment Examples

This directory contains **5 complete, runnable examples** that can reproduce key experiments from the paper. Each example is self-contained with its own shell script and can be executed independently to verify the pipeline works correctly. We have added experiments for the MIDOGpp dataset as its the latest mitotic figure dataset and can be downloaded easily. The examples use the ViT-S model because it is available to everyone and is computationally more feasible. If you want to try out LoRA fine-tuning with the foundation models, make sure you have access to them via HuggingFace and that it is supported in `src/classifier.py`. 

## Overview of Examples

| Example | Dataset | Method | Training Sizes | No. repititions | Purpose | Runtime |
|---------|---------|--------|----------------|-----------------|---------|---------|
| **`01_MIDOG2022_vit_mini_example_linear_probing`** | MIDOG2022 (debug subset) | **Linear Probing** | 100% | 1 | **Quick sanity check** | ~Minutes |
| **`02_MIDOG2022_vit_full_example_linear_probing`** | MIDOG2022 (full training) | **Linear Probing** | 0.1%, 1%, 10%, 100% | 5 | **Full linear probing experiment** | ~Hours |
| **`03_MIDOGpp_vit_full_example_linear_probing`** | **MIDOGpp** | **Linear Probing** | 0.1%, 1%, 10%, 100% | 5 | **Full linear probing experiment** | ~Hours |
| **`04_MIDOGpp_vit_example_lora`** | **MIDOGpp** | **LoRA Fine-tuning**  | 0.1%, 1%, 10%, 100% | 5 | **Standard LoRA fine-tuning** | **Long** |
| **`05_MIDOGpp_vit_example_cross_domain_lora`** | **MIDOGpp** | **LoRA Cross-Domain** | Full dataset | 5 | **Tumor-specific LoRA fine-tuning** | **Longest** |

## Quick Start

Each example follows this pattern:

1. **Make executable**: `chmod +x run_*.sh`
2. **Run**: `./run_*.sh`
3. **Check results**: Look for `results.pkl` in the output directory

More detailed descriptions can be found at each subfolder. 


## Prerequisites

- Check the prerequisited at the main page
- Download the required datasets using the google-drive link or the python scripts


## Expected Outputs

Each example produces a single `results.pkl` file containing aggregated metrics suitable for plotting/comparison.

