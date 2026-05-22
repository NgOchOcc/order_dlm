set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/hdo_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

# Configuration
MODEL_NAME=${MODEL_NAME:-"GSAI-ML/LLaDA-8B-Base"}
VERIFIER_TYPE=${VERIFIER_TYPE:-"qwen_vllm"}
GPU_LLADA=${GPU_LLADA:-0}
GPU_QWEN=${GPU_QWEN:-1}
MAX_SAMPLES=${MAX_SAMPLES:-500}
STEPS=${STEPS:-256}
GEN_LENGTH=${GEN_LENGTH:-1024}
BLOCK_LENGTH=${BLOCK_LENGTH:-1024}
BETA=${BETA:-1.0}
CANDIDATE_BRANCHING=${CANDIDATE_BRANCHING:-8}
BACKUP_WIDTH=${BACKUP_WIDTH:-8}
MAX_BACKUP_DEPTH=${MAX_BACKUP_DEPTH:-8}
OUTPUT_FILE="${OUTPUT_DIR}/results.json"

CUDA_VISIBLE_DEVICES=${GPU_LLADA} python eval_hdo.py \
    --model_name "${MODEL_NAME}" \
    --verifier_type "${VERIFIER_TYPE}" \
    --verifier_gpu ${GPU_QWEN} \
    --max_samples ${MAX_SAMPLES} \
    --steps ${STEPS} \
    --gen_length ${GEN_LENGTH} \
    --block_length ${BLOCK_LENGTH} \
    --beta ${BETA} \
    --candidate_branching ${CANDIDATE_BRANCHING} \
    --backup_width ${BACKUP_WIDTH} \
    --max_backup_depth ${MAX_BACKUP_DEPTH} \
    --output_file "${OUTPUT_FILE}" \
    2>&1 | tee "${OUTPUT_DIR}/llada_hbo.txt"
