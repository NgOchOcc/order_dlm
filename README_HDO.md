# HDO-DLM: Harmonic Doob Ordering for Diffusion Language Models

Implementation of **HDO-DLM** (Harmonic Doob Ordering for Diffusion Language Models), a test-time scaling algorithm for diffusion language models based on the Doob h-transform.

## Overview

HDO-DLM implements the key insight from the proposal:

> **Do not denoise the easiest token. Denoise the edge with the largest harmonic future-reward advantage.**

Instead of using confidence-based denoising order (which selects high-confidence tokens), HDO-DLM uses the Doob h-transform to select denoising transitions based on their expected future reward potential.

### Key Components

1. **Backward Information Function**: `h_t(x_t) = E[exp(βR(X_0)) | X_t = x_t]`
2. **Doob Transform**: `K*_t(x_{t-1} | x_t) = K_t(x_{t-1} | x_t) × h_{t-1}(x_{t-1}) / h_t(x_t)`
3. **Bellman Calibration**: Iterative refinement of h estimates to satisfy backward harmonic equation
4. **Adaptive Compute**: Allocate more backup iterations where martingale residual is high

## Installation

### Requirements

```bash
pip install torch transformers numpy
pip install math-verify  # For answer verification
```

### Optional: Qwen2.5-Math-PRM-7B

For full verifier support (recommended for best results):

```bash
# Qwen2.5-Math-PRM-7B will be downloaded automatically when using --verifier_type qwen
```

## Quick Start

### Basic Usage

```bash
# Run HDO-DLM on Math500 with lightweight verifier (fast prototyping)
python eval_hdo.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --verifier_type lightweight \
    --max_samples 10 \
    --steps 128 \
    --gen_length 256 \
    --output_file hdo_results.json

# Run with Qwen2.5-Math-PRM verifier (better quality)
python eval_hdo.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --verifier_type qwen \
    --max_samples 10 \
    --output_file hdo_qwen_results.json
```

### Ablation Study: No Bellman Calibration

```bash
# Test clean prediction only (no Bellman backup)
python eval_hdo.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --no_calibration \
    --output_file hdo_no_calibration.json
```

This ablation corresponds to Section 16, Ablation 1 in the proposal: using `h = exp(βV(x̂_0))` only.

### Baseline Comparison

```bash
# Run baseline confidence-based LLaDA
python eval_llada.py \
    --model_name GSAI-ML/LLaDA-8B-Base \
    --max_samples 10 \
    --output_file llada_baseline.json
```

## HDO-DLM Parameters

### Core Algorithm Parameters

- `--beta` (default: 1.0): Reward temperature β
  - Higher values emphasize high-reward completions more strongly
  - Lower values make the distribution more uniform

- `--candidate_branching` (default: 4): Number of candidate children B
  - More candidates → better edge selection but higher compute
  - Section 10.1: "candidate branching B"

- `--backup_width` (default: 4): Number of rollout children M per Bellman backup
  - More rollouts → better h estimate but higher compute
  - Section 9.3: "M children" for Bellman backup

- `--max_backup_depth` (default: 4): Maximum Bellman iterations D_max
  - Deeper backups → more accurate h but higher compute
  - Adaptive: actual depth depends on residual

- `--residual_threshold` (default: 0.1): Threshold ε for early stopping
  - Lower threshold → more accurate h but slower
  - Section 9.3: "if E_hat < eps: break"

- `--exploration_temp` (default: 0.0): Temperature τ for edge selection
  - 0.0 = greedy (select best edge)
  - Higher values add exploration/diversity

### Generation Parameters

- `--steps` (default: 128): Number of denoising steps
- `--gen_length` (default: 256): Length of generation
- `--block_length` (default: 32): Block size for block-level denoising

## Architecture

```
order_dlm/
├── hdo_dlm.py              # Main HDO-DLM algorithm (Algorithm 1)
├── harmonic_estimator.py   # Backward information function h_t estimation
│                          # - Clean prediction initialization
│                          # - Bellman-calibrated backup
│                          # - Adaptive depth scheduling
├── verifier.py             # Verifier/reward model wrappers
│                          # - Qwen2.5-Math-PRM-7B
│                          # - Lightweight heuristic verifier
├── eval_hdo.py             # Evaluation script for Math500
├── eval_llada.py           # Baseline LLaDA evaluation
└── README_HDO.md           # This file
```

## Algorithm Flow

From Algorithm 1 in the proposal:

```
For each denoising step t:
  1. Candidate Expansion:
     - Sample B candidate children y_b ~ K_t(· | x_t)
     - Use mixed strategy: confidence + entropy + random

  2. Harmonic Evaluation:
     - Initialize: ell_0(y_b) = β * V(CleanPredict(y_b))
     - Bellman Calibration (adaptive depth):
       for d = 1, ..., D_max:
         - Sample M rollout children z_m ~ K_{t-1}(· | y_b)
         - ell_d(y_b) = logmeanexp_m ell_{d-1}(z_m)
         - Compute residual E_hat
         - if E_hat < eps: break

  3. Doob Edge Selection:
     - score_b = log K_t(y_b | x_t) + ell_d(y_b)
     - Select y* by softmax(score_b / tau) or greedy

  4. Update State:
     - x_{t-1} = y*
```

## Expected Performance

From Section 17.1 of the proposal, HDO-DLM is expected to help most when:

1. **Final correctness depends on low-confidence but high-impact tokens**
   - Math reasoning: operators, variables, intermediate numbers
   - Code: structural tokens, logic connectives

2. **Confidence schedules commit filler tokens too early**
   - Easy but irrelevant tokens get revealed first
   - Decision-critical tokens remain masked longer

3. **Verifier lookahead benefits from temporal calibration**
   - Clean prediction alone may be biased
   - Bellman backup enforces consistency

## Computational Cost

From Section 14.3:

```
NFE ≈ N × T × (1 + B + B × M × D)
```

Where:
- N = num_particles (default: 1)
- T = steps (default: 128)
- B = candidate_branching (default: 4)
- M = backup_width (default: 4)
- D = avg backup depth (adaptive, typically 2-4)

For default settings:
- Baseline (confidence): ~128 NFE
- HDO-DLM (no calibration): ~640 NFE (5× baseline)
- HDO-DLM (full): ~2,560-5,120 NFE (20-40× baseline)

**Trade-off**: More compute → better accuracy via improved harmonic estimates.

## Comparison to Prior Methods

| Method | Main Scaling Axis | Transition Modification |
|--------|------------------|------------------------|
| **PG-DLM** | Trajectory refinement iterations | Conditional SMC on full trajectories |
| **S³** | Frontier expansion & resampling | Verifier-guided frontier scores |
| **HDO-DLM** | Harmonic backup accuracy | Direct h-ratio edge weighting |

From Section 11:

- **vs S³**: S³ uses verifier lookahead as frontier score. HDO-DLM treats it as approximate h and enforces Bellman equation.
- **vs PG-DLM**: PG-DLM refines trajectories. HDO-DLM changes individual transitions via Doob transform.
- **vs Twisted SMC**: Similar mathematical principle, but applied to DLM mask lattice for denoising-order rule.

## Key Experiments

### Most Important Experiment (Section 15.4)

> Replace S³'s raw clean-prediction frontier score with Bellman-calibrated h estimate and use the resulting Doob edge score for denoising order. At matched NFE, measure whether performance improves and whether improvement correlates with reduced martingale residual.

This can be approximated by comparing:

```bash
# HDO-DLM with calibration (our method)
python eval_hdo.py --use_calibration

# HDO-DLM without calibration (similar to S³ lookahead)
python eval_hdo.py --no_calibration
```

### Ablation Studies (Section 16)

1. **No Bellman backup**: `--no_calibration`
2. **Backup depth**: `--max_backup_depth 0/1/2/4/8`
3. **Candidate branching**: `--candidate_branching 1/2/4/8`
4. **Backup width**: `--backup_width 1/2/4/8`
5. **Exploration**: `--exploration_temp 0.0/0.5/1.0`

## Troubleshooting

### Out of Memory

Reduce computational cost:
```bash
python eval_hdo.py \
    --candidate_branching 2 \
    --backup_width 2 \
    --max_backup_depth 2
```

### Slow Evaluation

Use lightweight verifier for prototyping:
```bash
python eval_hdo.py --verifier_type lightweight
```

Or reduce sample size:
```bash
python eval_hdo.py --max_samples 10
```

### Poor Results

Try increasing reward temperature:
```bash
python eval_hdo.py --beta 2.0
```

Or use Qwen verifier instead of lightweight:
```bash
python eval_hdo.py --verifier_type qwen
```

## Citation

If you use this implementation, please cite the HDO-DLM proposal:

```bibtex
@article{hdodlm2026,
  title={The Right Denoising Order Is a Doob Transform: Harmonic Test-Time Scaling for Diffusion Language Models},
  author={[Authors]},
  year={2026}
}
```

## License

This implementation is for research purposes.

## Contact

For questions or issues, please open an issue in the repository.
