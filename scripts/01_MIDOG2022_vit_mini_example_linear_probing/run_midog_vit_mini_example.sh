# Mini example: ViT-S feature extraction + linear probing on MIDOG2022 debug subset.

echo "=== Step 1/2: Extracting ViT_S features on MIDOG2022 debug subset ==="
uv run ../../extract_features.py \
  --path_to_csv_file ../../databases/MIDOG2022_training_debug_tiff.csv \
  --image_dir ../../MIDOG2022/images \
  --out_path mini_example \
  --batch_size 32 \
  --num_workers 4 \
  --model ViT_S

echo "=== Step 2/2: Running linear probing on extracted ViT_S features ==="
uv run ../../linear_probing.py \
  --path_to_features mini_example/ViT_S_features.pkl \
  --model_name ViT_S \
  --save_dir mini_example \
  --train_sizes 1 \
  --seeds 43

echo "Done. Results saved to mini_example/ViT_S.pkl"