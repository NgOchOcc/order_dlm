"""
Inference and Evaluation Script for LLaDA Diffusion Language Model on Math500 Dataset
Uses the proper diffusion sampling method for LLaDA models
Uses math_verify library for accurate mathematical equivalence checking
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import json
import re
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import os

try:
    from math_verify import math_equal
    MATH_VERIFY_AVAILABLE = True
    print("math_verify loaded successfully")
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not available. Using string comparison fallback.")


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_llada(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128,
                   temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
                   logits_eos_inf=False, confidence_eos_eot_inf=False):
    """
    Generate text using LLaDA diffusion sampling

    Args:
        model: Mask predictor (LLaDA model).
        prompt: A tensor of shape (B, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


class LLaDAMath500Evaluator:
    """Evaluator for LLaDA Diffusion Model on Math500 dataset"""

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Base", device: str = "auto"):
        """
        Initialize the evaluator with LLaDA model and tokenizer

        Args:
            model_name: HuggingFace model name (LLaDA-8B-Base or LLaDA-8B-Instruct)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        print(f"Loading LLaDA model: {model_name}")
        self.model_name = model_name
        self.is_instruct = 'Instruct' in model_name

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load tokenizer and model with AutoModel (not AutoModelForCausalLM for LLaDA)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # LLaDA uses left padding for batch generation
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        # Verify mask token ID
        self.mask_id = 126336
        if self.tokenizer.pad_token_id == self.mask_id:
            raise ValueError("Padding token ID conflicts with mask token ID!")

        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        print("LLaDA model loaded successfully!")

    def load_math500_dataset(self, dataset_path: str = None):
        """
        Load Math500 dataset

        Args:
            dataset_path: Local path to dataset or HuggingFace dataset name
        """
        print("Loading Math500 dataset...")

        if dataset_path and os.path.exists(dataset_path):
            # Load from local file
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        else:
            # Try loading from HuggingFace or use a common math benchmark
            try:
                # Math500 might be available as MATH dataset subset
                dataset = load_dataset("hendrycks/competition_math", split="test")
                # Take first 500 samples
                self.dataset = [sample for sample in dataset.select(range(min(500, len(dataset))))]
            except:
                print("Warning: Could not load Math500. Using GSM8K as alternative.")
                dataset = load_dataset("gsm8k", "main", split="test")
                self.dataset = [sample for sample in dataset.select(range(min(500, len(dataset))))]

        print(f"Loaded {len(self.dataset)} problems")
        return self.dataset

    def create_prompt(self, problem: str) -> str:
        """
        Create prompt for the LLaDA model

        Args:
            problem: Math problem text
        """
        if self.is_instruct:
            # For Instruct model, use chat template
            messages = [{"role": "user", "content": problem}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            # For Base model, simple prompt
            prompt = f"{problem}\n\nAnswer:"

        return prompt

    def generate_answer(self, problem: str, steps: int = 128, gen_length: int = 256,
                       block_length: int = 32, temperature: float = 0., cfg_scale: float = 0.,
                       remasking: str = 'low_confidence') -> str:
        """
        Generate answer using LLaDA diffusion sampling

        Args:
            problem: Math problem text
            steps: Number of diffusion steps
            gen_length: Maximum generation length
            block_length: Block length for semi-autoregressive generation
            temperature: Sampling temperature (0 for greedy)
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
        """
        prompt = self.create_prompt(problem)

        # Tokenize
        encoded = self.tokenizer(
            [prompt],
            add_special_tokens=False,
            padding=False,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Generate with diffusion sampling
        output_ids = generate_llada(
            self.model,
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=self.mask_id
        )

        # Decode only the generated part
        generated_text = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        return generated_text.strip()

    def extract_answer(self, text: str) -> str:
        """
        Extract final answer from generated text

        Args:
            text: Generated text containing the solution
        """
        # Common patterns for final answers
        patterns = [
            r"(?:final answer|answer|solution)(?:\s+is)?:?\s*[\$]?([^$\n]+)[\$]?",
            r"\\boxed\{([^}]+)\}",
            r"####\s*(.+)",
            r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # If no pattern matches, try to find the last number
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return text.strip()

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (fallback method)"""
        answer = answer.strip()
        answer = answer.replace(",", "").replace("$", "").replace("\\", "")
        answer = answer.lower()
        return answer

    def check_answer_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth using math_verify

        Args:
            predicted: Predicted answer string
            ground_truth: Ground truth answer string

        Returns:
            bool: True if answers are mathematically equivalent
        """
        if MATH_VERIFY_AVAILABLE:
            try:
                is_correct = math_equal(predicted, ground_truth)
                return is_correct
            except Exception as e:
                # Fallback to string comparison if math_verify fails
                return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)
        else:
            return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)

    def evaluate_dataset(self, output_file: str = "results.json", max_samples: int = None,
                        steps: int = 128, gen_length: int = 256, block_length: int = 32,
                        temperature: float = 0., cfg_scale: float = 0.) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset

        Args:
            output_file: File to save results
            max_samples: Maximum number of samples to evaluate (None for all)
            steps: Number of diffusion steps
            gen_length: Maximum generation length
            block_length: Block length for generation
            temperature: Sampling temperature
            cfg_scale: Classifier-free guidance scale
        """
        results = []
        correct = 0
        total = 0

        samples = self.dataset[:max_samples] if max_samples else self.dataset

        print(f"\nEvaluating on {len(samples)} samples...")
        print(f"Generation config: steps={steps}, gen_length={gen_length}, block_length={block_length}")
        print(f"temperature={temperature}, cfg_scale={cfg_scale}\n")

        for idx, sample in enumerate(tqdm(samples)):
            # Extract problem and answer based on dataset format
            if isinstance(sample, dict):
                if 'problem' in sample:
                    problem = sample['problem']
                    ground_truth = sample.get('solution', sample.get('answer', ''))
                elif 'question' in sample:
                    problem = sample['question']
                    ground_truth = sample.get('answer', '')
                else:
                    continue
            else:
                continue

            # Generate answer
            try:
                generated = self.generate_answer(
                    problem,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale
                )
                predicted_answer = self.extract_answer(generated)

                # Extract ground truth answer
                if isinstance(ground_truth, str):
                    gt_answer = self.extract_answer(ground_truth)
                else:
                    gt_answer = str(ground_truth)

                # Compare answers using math_verify
                is_correct = self.check_answer_correct(predicted_answer, gt_answer)

                if is_correct:
                    correct += 1
                total += 1

                # Store result
                result = {
                    'index': idx,
                    'problem': problem,
                    'ground_truth': gt_answer,
                    'generated_solution': generated,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct
                }
                results.append(result)

                # Print progress every 50 samples
                if (idx + 1) % 50 == 0:
                    current_acc = (correct / total) * 100
                    print(f"\nProgress: {idx + 1}/{len(samples)} | Accuracy: {current_acc:.2f}%")

            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                continue

        # Calculate final metrics
        accuracy = (correct / total * 100) if total > 0 else 0

        metrics = {
            'model': self.model_name,
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'generation_config': {
                'steps': steps,
                'gen_length': gen_length,
                'block_length': block_length,
                'temperature': temperature,
                'cfg_scale': cfg_scale
            },
            'results': results
        }

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*50}")
        print(f"Evaluation Complete!")
        print(f"{'='*50}")
        print(f"Total Samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Results saved to: {output_file}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA Diffusion Model on Math500")
    parser.add_argument(
        "--model_name",
        type=str,
        default="GSAI-ML/LLaDA-8B-Base",
        help="HuggingFace model name (LLaDA-8B-Base or LLaDA-8B-Instruct)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to Math500 dataset (if local)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="llada_math500_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Number of diffusion sampling steps"
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        default=256,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=32,
        help="Block length for semi-autoregressive generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = LLaDAMath500Evaluator(
        model_name=args.model_name,
        device=args.device
    )

    # Load dataset
    evaluator.load_math500_dataset(args.dataset_path)

    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        output_file=args.output_file,
        max_samples=args.max_samples,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale
    )


if __name__ == "__main__":
    main()
