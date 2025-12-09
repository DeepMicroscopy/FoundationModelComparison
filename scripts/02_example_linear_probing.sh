# Small example script that runs linear_probing.py via `uv run` on example ViT_S
# features. It evaluates linear probing at two training sizes (10% and 100%) with
# two random seeds (42 and 43), using example/linear_probing/features/ViT_S.pkl and saving the
# results under example/linear_probing/results/ViT_S.pkl.

uv run linear_probing.py --path_to_features example/features/ViT_S.pkl --model_name ViT_S --save_dir example/linear_probing/results --train_sizes 0.1 1 --seeds 42 43