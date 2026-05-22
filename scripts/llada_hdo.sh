#!/bin/bash

# HDO-DLM Runner - Simple execution script
# Usage: ./run_hdo.sh

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/hdo_${TIMESTAMP}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HDO-DLM Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Default parameters (can be overridden by environment variables)
MODEL_NAME=${MODEL_NAME:-"GSAI-ML/LLaDA-8B-Base"}
VERIFIER_TYPE=${VERIFIER_TYPE:-"lightweight"}
MAX_SAMPLES=${MAX_SAMPLES:-50}
STEPS=${STEPS:-128}
GEN_LENGTH=${GEN_LENGTH:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
BETA=${BETA:-1.0}
CANDIDATE_BRANCHING=${CANDIDATE_BRANCHING:-4}
BACKUP_WIDTH=${BACKUP_WIDTH:-4}
MAX_BACKUP_DEPTH=${MAX_BACKUP_DEPTH:-4}
RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD:-0.1}
EXPLORATION_TEMP=${EXPLORATION_TEMP:-0.0}

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Verifier: ${VERIFIER_TYPE}"
echo "  Max samples: ${MAX_SAMPLES}"
echo "  Steps: ${STEPS}"
echo "  Gen length: ${GEN_LENGTH}"
echo "  Block length: ${BLOCK_LENGTH}"
echo ""
echo "HDO-DLM Parameters:"
echo "  Beta (β): ${BETA}"
echo "  Candidate branching (B): ${CANDIDATE_BRANCHING}"
echo "  Backup width (M): ${BACKUP_WIDTH}"
echo "  Max backup depth (D_max): ${MAX_BACKUP_DEPTH}"
echo "  Residual threshold (ε): ${RESIDUAL_THRESHOLD}"
echo "  Exploration temp (τ): ${EXPLORATION_TEMP}"
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/hdo_results.json"
LOG_FILE="${OUTPUT_DIR}/hdo_results.log"

echo -e "${YELLOW}Running HDO-DLM...${NC}"
echo "Output: ${OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
echo ""

python eval_hdo.py \
    --model_name "${MODEL_NAME}" \
    --verifier_type "${VERIFIER_TYPE}" \
    --max_samples ${MAX_SAMPLES} \
    --steps ${STEPS} \
    --gen_length ${GEN_LENGTH} \
    --block_length ${BLOCK_LENGTH} \
    --beta ${BETA} \
    --candidate_branching ${CANDIDATE_BRANCHING} \
    --backup_width ${BACKUP_WIDTH} \
    --max_backup_depth ${MAX_BACKUP_DEPTH} \
    --residual_threshold ${RESIDUAL_THRESHOLD} \
    --exploration_temp ${EXPLORATION_TEMP} \
    --output_file "${OUTPUT_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HDO-DLM Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""

# Display summary
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Summary:"
    accuracy=$(python -c "import json; data=json.load(open('${OUTPUT_FILE}')); print(f\"{data.get('accuracy', 0):.2f}%\")" 2>/dev/null || echo "N/A")
    total=$(python -c "import json; data=json.load(open('${OUTPUT_FILE}')); print(data.get('total', 0))" 2>/dev/null || echo "N/A")
    correct=$(python -c "import json; data=json.load(open('${OUTPUT_FILE}')); print(data.get('correct', 0))" 2>/dev/null || echo "N/A")
    avg_nfe=$(python -c "import json; data=json.load(open('${OUTPUT_FILE}')); print(f\"{data.get('avg_nfe', 0):.1f}\")" 2>/dev/null || echo "N/A")
    avg_time=$(python -c "import json; data=json.load(open('${OUTPUT_FILE}')); print(f\"{data.get('avg_time_per_sample', 0):.2f}s\")" 2>/dev/null || echo "N/A")

    echo "  Accuracy: ${accuracy} (${correct}/${total})"
    echo "  Avg NFE: ${avg_nfe}"
    echo "  Avg time/sample: ${avg_time}"
fi

echo ""
echo "Check ${OUTPUT_FILE} for detailed results"
