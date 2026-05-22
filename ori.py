python eval_llada.py \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --dataset_path dataset/test_math500.jsonl \
    --steps 256 \
    --gen_length 1024 \
    --block_length 256 \
    --max_samples 100 \
    --output_file output/math500.json