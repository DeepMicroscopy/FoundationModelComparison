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
uv run ../../calc_metrics.py \
    --model_result_dirs ViT_S_LoRA \
    --exp_name ViT_S_LoRA \
    --path_to_results experiment \
    --num_folds 5

echo "Done. Results saved to experiment/ViT_S_LoRA/results.pkl"