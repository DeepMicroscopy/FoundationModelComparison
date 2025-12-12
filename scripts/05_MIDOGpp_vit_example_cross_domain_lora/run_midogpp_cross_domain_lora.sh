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
