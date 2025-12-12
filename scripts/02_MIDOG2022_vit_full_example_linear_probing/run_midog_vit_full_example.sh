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
  --train_sizes 0.001 0.01 0.1 1\
  --seeds 42 43 44 45 46


echo "Done. Results saved to full_example/ViT_S.pkl"