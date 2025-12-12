# Benchmarking Foundation Models for Mitotic Figure Classification

This repository hosts the code that was used for our paper [Benchmarking Foundation Models for Mitotic Figure Classification](https://arxiv.org/abs/2508.04441).

## Abstract 
The performance of deep learning models is known to scale with data quantity and diversity. In pathology, as in many other medical imaging domains, the availability of labeled images for a specific task is often limited. Self-supervised learning techniques have enabled the use of vast amounts of unlabeled data to train large-scale neural networks, i.e., foundation models, that can address the limited data problem by providing semantically rich feature vectors that can generalize well to new tasks with minimal training effort increasing model performance and robustness. In this work, we investigate the use of foundation models for mitotic figure classification. The mitotic count, which can be derived from this classification task, is an independent prognostic marker for specific tumors and part of certain tumor grading systems. In particular, we investigate the data scaling laws on multiple current foundation models and evaluate their robustness to unseen tumor domains. Next to the commonly used linear probing paradigm, we also adapt the models using low-rank adaptation (LoRA) of their attention mechanisms. We compare all models against end-to-end-trained baselines, both CNNs and Vision Transformers. Our results demonstrate that LoRA-adapted foundation models provide superior performance to those adapted with standard linear probing, reaching performance levels close to 100 % data availability with only 10 % of training data. Furthermore, LoRA-adaptation of the most recent foundation models almost closes the out-of-domain performance gap when evaluated on unseen tumor domains. However, full fine-tuning of traditional architectures still yields competitive performance.


## Prerequisties
- Python 3.10 or higher 
- CUDA compatible GPU (recommended for training)
- [uv](https://github.com/astral-sh/uv) package manager
- If you want to reproduce some of our results using the foudnation models such as Virchow2, you need to make sure that you have access to these models via [HuggingFace](https://huggingface.co/).

## Installation

We recommend using `uv` for managing dependencies and the Python environment.

### Installing uv
If you don't have `uv` installed, you can install it using:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, you can install it via pip:
```bash
pip install uv
```

### Setting up the repository
1. Clone the repository:
```bash
git clone git@github.com:DeepMicroscopy/FoundationModelComparison.git
cd FoundationModelComparison
```

2. Install dependices using `uv`:
```bash
uv sync
```
This will create a virtual environment and install all required dependencies as specificied in `pyproject.toml`.

3. Activate the virtual environment:
```bash 
# The virtual enviroment is automativally created by uv
# To run comandas in the environment, prefix them with `uv run`
# For example:
uv run train_classifier.py arg1 arg2 arg3
```

## Code Structure
The repository contins the following main scripts for reproducing the experiments:
- `train_classifier.py`: Train different models including CNNs and Vision Transformers on multiple training data fractions and random seeds optionally with LoRA fine-tuning.
- `train_classifier_tumor_specific.py`: Train tumor specific models for cross-domain analysis. Supports LoRA fine-tuning and multiple training runs. 
- `extract_features.py`: Extract feature represenations from certain models including foundation models. 
- `linear_probing.py`: Perform linear probing on extracted features from foundation models. 
- `calc_metrics.py`: Calculate and aggregate performance metrics from experiments. 
- `download_CCMCT.py`: Downloads the [MITOS_WSI_CCMCT](https://github.com/DeepMicroscopy/MITOS_WSI_CCMCT/tree/master) dataset. 
- `download_MIDOGpp.py`: Downloads the [MIDOG++](https://github.com/DeepMicroscopy/MIDOGpp) dataset.
- `download_MIDOG2022.py`: Downloads the [MIDOG2022](https://www.sciencedirect.com/science/article/pii/S136184152400080X) dataset.
- `databases/`: Contains some database files that were used in our paper. 

## Getting the data
When you setup the environment as described above you can download the images of the MIDOGpp dataset, the MIDOG2022 dataset or the CCMCT dataset by one of the following commands:

```bash
# Download the MIDOG++
uv run download_MIDOGpp.py
```

```bash
# Download the CCMCT dataset
uv run download_CCMCT.py
```

```bash
# Download the MIDOG 2022 (png) dataset (not recommended)
uv run download_MIDOG2022.py
```

The experiments in the paper were performed on the .tiff version of the MIDOG 2022 dataset. The .tiff dataset can be downloaded via this google-drive [link](https://drive.google.com/drive/folders/1P73g1xg8jw_JGLJaDFQDnxwQA7ROVykA). If you download the entire folder to `MIDOG2022` you can use the this [script](scripts/01_example_extract_features.sh) to verify that the code runs. 


## Usage examples
We have provided some database files that were used for our experiments at `databases`. 


## Citation
If you use this code in your research, please cite our paper:
```bibtex
@misc{ammeling2025benchmarkingfoundationmodelsmitotic,
      title={Benchmarking Foundation Models for Mitotic Figure Classification}, 
      author={Jonas Ammeling and Jonathan Ganz and Emely Rosbach and Ludwig Lausser and Christof A. Bertram and Katharina Breininger and Marc Aubreville},
      year={2025},
      eprint={2508.04441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.04441}, 
}
```