CUDA_VISIBLE_DEVICES=0 python eval_llada.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --dataset_path dataset/test500.jsonl \
    --steps 1024 \
    --gen_length 1024 \
    --block_length 1024 \
    --max_samples 500 \
    --output_file output/math500.json