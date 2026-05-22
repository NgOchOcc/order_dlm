"""LLaDA Diffusion Model Evaluation on Math Datasets"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import re
import argparse
import os
import time
from datetime import datetime

try:
    from math_verify import math_equal
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not available, using string comparison")


def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise for categorical sampling (float64 for precision)"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Compute number of tokens to transfer at each diffusion step"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_llada(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128,
                   temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """LLaDA diffusion sampling"""
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length),
                                    dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length:
                              prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
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

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class LLaDAEvaluator:
    """LLaDA Model Evaluator"""

    def __init__(self, model_name="GSAI-ML/LLaDA-8B-Base", device="auto"):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.is_instruct = 'Instruct' in model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.mask_id = 126336

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token_id == self.mask_id:
            raise ValueError("Padding token ID conflicts with mask token ID")

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device).eval()

        print(f"Model loaded on {self.device}")

    def load_dataset(self, dataset_path=None):
        """Load dataset from jsonl file"""
        if not dataset_path:
            dataset_path = "dataset/test500.jsonl"

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                self.dataset.append(json.loads(line))

        print(f"Loaded {len(self.dataset)} problems from {dataset_path}")

    def create_prompt(self, problem):
        """Create prompt with instruction to use \\boxed{} format"""
        if self.is_instruct:
            instruction = "Solve the following math problem step by step. Put your final answer in \\boxed{}."
            messages = [{"role": "user", "content": f"{instruction}\n\n{problem}"}]
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            return f"Problem: {problem}\n\nSolve the problem step by step and put your final answer in \\boxed{{}}.\n\nSolution:"

    def generate(self, problem, steps=128, gen_length=256, block_length=32,
                 temperature=0., cfg_scale=0., remasking='low_confidence'):
        """Generate answer using diffusion sampling"""
        prompt = self.create_prompt(problem)
        encoded = self.tokenizer([prompt], add_special_tokens=False, padding=False, return_tensors="pt")
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        output_ids = generate_llada(
            self.model, input_ids, attention_mask=attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=temperature, cfg_scale=cfg_scale, remasking=remasking, mask_id=self.mask_id
        )

        return self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    def extract_answer(self, text):
        """Extract answer from \\boxed{} or other patterns"""
        # First try to extract from \boxed{}
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try other patterns
        patterns = [
            r"####\s*(.+)",
            r"(?:final answer|answer|solution)(?:\s+is)?:?\s*[\$]?([^$\n]+)[\$]?",
            r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Fallback: find last number
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else text.strip()

    def check_correct(self, predicted, ground_truth):
        """Check if answer is correct using math_verify or string comparison"""
        if MATH_VERIFY_AVAILABLE:
            try:
                return math_equal(predicted, ground_truth)
            except:
                pass

        # Fallback
        norm = lambda x: x.strip().replace(",", "").replace("$", "").replace("\\", "").lower()
        return norm(predicted) == norm(ground_truth)

    def evaluate(self, output_file="results.json", max_samples=None, steps=128,
                 gen_length=256, block_length=32, temperature=0., cfg_scale=0.):
        """Evaluate on dataset"""
        results = []
        correct = total = 0
        samples = self.dataset[:max_samples] if max_samples else self.dataset

        start_time = time.time()

        print(f"\nEvaluating {len(samples)} samples")
        print(f"Config: steps={steps}, gen_length={gen_length}, block_length={block_length}, "
              f"temp={temperature}, cfg={cfg_scale}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print("="*100)

        for idx, sample in enumerate(samples):
            problem = sample.get('problem')
            ground_truth = sample.get('answer', '')
            if not problem:
                continue

            try:
                sample_start = time.time()

                generated = self.generate(problem, steps=steps, gen_length=gen_length,
                                         block_length=block_length, temperature=temperature, cfg_scale=cfg_scale)
                predicted = self.extract_answer(generated)
                gt_answer = self.extract_answer(ground_truth) if isinstance(ground_truth, str) else str(ground_truth)
                is_correct = self.check_correct(predicted, gt_answer)

                if is_correct:
                    correct += 1
                total += 1

                sample_time = time.time() - sample_start
                total_time = time.time() - start_time
                acc = correct / total * 100

                # Print detailed info for each sample
                status = "✓" if is_correct else "✗"
                print(f"{status} [{idx+1}/{len(samples)}] | Acc: {acc:.2f}% | Time: {format_time(sample_time)} | "
                      f"Total: {format_time(total_time)}")
                print(f"  Pred: {predicted}")
                print(f"  GT:   {gt_answer}")
                print("-"*100)

                results.append({
                    'index': idx,
                    'problem': problem,
                    'ground_truth': gt_answer,
                    'generated': generated,
                    'predicted': predicted,
                    'correct': is_correct,
                    'time': sample_time
                })

            except Exception as e:
                print(f"✗ Error at sample {idx}: {e}")
                print("-"*100)
                continue

        total_time = time.time() - start_time
        accuracy = (correct / total * 100) if total > 0 else 0

        metrics = {
            'model': self.model_name,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'total_time': total_time,
            'avg_time_per_sample': total_time / total if total > 0 else 0,
            'config': {'steps': steps, 'gen_length': gen_length, 'block_length': block_length,
                      'temperature': temperature, 'cfg_scale': cfg_scale},
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print("="*100)
        print(f"Final Results: {correct}/{total} = {accuracy:.2f}%")
        print(f"Total time: {format_time(total_time)}")
        print(f"Avg time/sample: {total_time/total:.2f}s" if total > 0 else "N/A")
        print(f"Saved to: {output_file}")
        print("="*100)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="LLaDA Evaluation on Math Datasets")
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--dataset_path", default="dataset/test500.jsonl")
    parser.add_argument("--output_file", default="results.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    args = parser.parse_args()

    evaluator = LLaDAEvaluator(model_name=args.model_name, device=args.device)
    evaluator.load_dataset(args.dataset_path)
    evaluator.evaluate(
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
