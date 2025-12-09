# Small example script that runs train_classifier.py via `uv run` to train a mitotic
# figure classification model. It uses the MIDOG2022_training_debug_png.csv
# dataset example, and saves checkpoints to example/train under the experiment 
# code ViT_S. The training is configured for the ViT_S model, with a batch size of 16, 
# data augmentation enabled,a learning rate scheduler, 10 epochs of patience for early stopping,
# and runs for 30 epochs across two training sizes (10% and 100%) with
# a single random seed (42).

uv run classifier.py --path_to_csv_file databases/MIDOG2022_training_debug_png.csv --image_dir MIDOG2022 --checkpoint_path example/train --exp_code ViT_S --scheduler --patience 10 --model_name ViT_S --batch_size 16 --augmentation --train_sizes 0.1 1 --seeds 42 --num_epochs 30