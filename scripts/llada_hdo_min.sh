mkdir -p results/

# Run HDO-DLM on 1 sample
echo "Running HDO-DLM..."
python eval_hdo.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --verifier_type qwen \
    --max_samples 500 \
    --steps 256 \
    --gen_length 1024 \
    --block_length 1024 \
    --beta 1.0 \
    --candidate_branching 4 \
    --backup_width 4 \
    --max_backup_depth 8 \
    --output_file results/hdo.json \
    --verbose
