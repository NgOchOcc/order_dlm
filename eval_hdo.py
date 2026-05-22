import torch
from transformers import AutoTokenizer, AutoModel
import json
import argparse
import os
import time
from datetime import datetime
import numpy as np

from hdo_dlm import HDODLM
from verifier import PRMScorer
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig
import re


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format"""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None


def verify_answer(predicted, ground_truth):
    """Verify answer using math_verify"""
    try:
        config = LatexExtractionConfig()
        pred_parsed = parse(predicted, extraction_config=config)
        gt_parsed = parse(ground_truth, extraction_config=config)
        result = verify(pred_parsed, gt_parsed)
        # Convert to boolean if needed
        return bool(result)
    except Exception as e:
        # Fallback to string comparison
        norm = lambda x: x.strip().replace(",", "").replace("$", "").replace("\\", "").lower()
        return norm(predicted) == norm(ground_truth)


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class HDOEvaluator:
    """HDO-DLM Evaluator for Math Datasets"""

    def __init__(
        self,
        model_name="GSAI-ML/LLaDA-8B-Base",
        verifier_gpu=1,
        device="auto",
        beta=1.0,
        num_particles=1,
        candidate_branching=4,
        backup_width=4,
        max_backup_depth=4,
        residual_threshold=0.1,
        exploration_temp=0.0,
    ):
        """
        Initialize HDO-DLM evaluator

        Args:
            model_name: Base DLM model
            verifier_type: "qwen" or "lightweight"
            device: Device to use
            beta: Reward temperature
            num_particles: Number of particles
            candidate_branching: Candidate children per state
            backup_width: Rollout children per backup
            max_backup_depth: Maximum Bellman iterations
            residual_threshold: Residual threshold for early stopping
            exploration_temp: Temperature for edge selection
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.is_instruct = 'Instruct' in model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.mask_id = 126336

        # Load tokenizer and model
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

        self.verifier = PRMScorer(gpu_id=verifier_gpu)

        # Initialize HDO-DLM
        self.hdo_dlm = HDODLM(
            model=self.model,
            tokenizer=self.tokenizer,
            verifier=self.verifier,
            mask_id=self.mask_id,
            beta=beta,
            num_particles=num_particles,
            candidate_branching=candidate_branching,
            backup_width=backup_width,
            max_backup_depth=max_backup_depth,
            residual_threshold=residual_threshold,
            exploration_temp=exploration_temp,
        )

    def load_dataset(self, dataset_path=None):
        if not dataset_path:
            dataset_path = "dataset/test500.jsonl"

        self.dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                self.dataset.append(json.loads(line))

    def create_prompt(self, problem):
        if self.is_instruct:
            instruction = "Solve the following math problem step by step. Put your final answer in \\boxed{}."
            messages = [{"role": "user", "content": f"{instruction}\n\n{problem}"}]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            return f"Problem: {problem}\n\nSolve the problem step by step and put your final answer in \\boxed{{}}.\n\nSolution:"

    def generate(
        self,
        problem,
        steps=128,
        gen_length=256,
        block_length=32,
        use_calibration=True,
        verbose=False
    ):
        """Generate using HDO-DLM"""
        prompt = self.create_prompt(problem)
        encoded = self.tokenizer([prompt], add_special_tokens=False, padding=False, return_tensors="pt")
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Generate with HDO-DLM
        output_ids, stats = self.hdo_dlm.generate(
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            use_calibration=use_calibration,
            verbose=verbose
        )

        generated = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return generated, stats

    def extract_answer(self, text):
        """Extract answer from generated text"""
        # First try boxed
        boxed = extract_boxed_answer(text)
        if boxed:
            return boxed

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

        # Fallback to last number
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else text.strip()

    def evaluate(
        self,
        output_file="hdo_results.json",
        max_samples=None,
        steps=128,
        gen_length=256,
        block_length=32,
        use_calibration=True,
        verbose=False
    ):
        """Evaluate HDO-DLM on Math500"""
        results = []
        correct = total = 0
        samples = self.dataset[:max_samples] if max_samples else self.dataset

        start_time = time.time()
        total_nfe = 0

        print(f"\nEvaluating HDO-DLM on {len(samples)} samples")
        print(f"Config: steps={steps}, gen_length={gen_length}, block_length={block_length}")
        print(f"HDO-DLM: beta={self.hdo_dlm.beta}, B={self.hdo_dlm.candidate_branching}, "
              f"M={self.hdo_dlm.backup_width}, D_max={self.hdo_dlm.max_backup_depth}")
        print(f"Calibration: {use_calibration}")
        print("=" * 100)

        for idx, sample in enumerate(samples):
            problem = sample.get('problem')
            ground_truth = sample.get('answer', '')
            if not problem:
                continue

            try:
                sample_start = time.time()

                generated, stats = self.generate(
                    problem,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    use_calibration=use_calibration,
                    verbose=verbose
                )

                predicted = self.extract_answer(generated)
                gt_answer = self.extract_answer(ground_truth) if isinstance(ground_truth, str) else str(ground_truth)
                is_correct = verify_answer(predicted, gt_answer)

                # Debug: Check if verify_answer is working correctly
                if predicted == gt_answer and not is_correct:
                    print(f"  WARNING: Predicted '{predicted}' == GT '{gt_answer}' but verify returned {is_correct}")

                if is_correct:
                    correct += 1
                total += 1
                total_nfe += stats['total_nfe']

                sample_time = time.time() - sample_start
                total_time = time.time() - start_time
                acc = correct / total * 100
                avg_nfe = total_nfe / total

                # Print detailed info
                status = "✓" if is_correct else "✗"
                print(f"{status} [{idx+1}/{len(samples)}] | Acc: {correct}/{total} = {acc:.2f}% | "
                      f"NFE: {stats['total_nfe']} (avg: {avg_nfe:.1f}) | "
                      f"Time: {format_time(sample_time)} | Total: {format_time(total_time)}")
                print(f"  Depth: {stats['avg_depth']:.2f} | Residual: {stats['avg_residual']:.4f}")
                print(f"  Pred: {predicted}")
                print(f"  GT:   {gt_answer}")
                print("-" * 100)

                results.append({
                    'index': idx,
                    'problem': problem,
                    'ground_truth': gt_answer,
                    'generated': generated,
                    'predicted': predicted,
                    'correct': is_correct,
                    'time': sample_time,
                    'nfe': stats['total_nfe'],
                    'avg_depth': stats['avg_depth'],
                    'avg_residual': stats['avg_residual']
                })

            except Exception as e:
                print(f"✗ Error at sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                print("-" * 100)
                continue

        total_time = time.time() - start_time
        accuracy = (correct / total * 100) if total > 0 else 0

        metrics = {
            'model': self.model_name,
            'method': 'HDO-DLM',
            'use_calibration': use_calibration,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'total_nfe': total_nfe,
            'avg_nfe': total_nfe / total if total > 0 else 0,
            'total_time': total_time,
            'avg_time_per_sample': total_time / total if total > 0 else 0,
            'config': {
                'steps': steps,
                'gen_length': gen_length,
                'block_length': block_length,
                'beta': self.hdo_dlm.beta,
                'candidate_branching': self.hdo_dlm.candidate_branching,
                'backup_width': self.hdo_dlm.backup_width,
                'max_backup_depth': self.hdo_dlm.max_backup_depth,
                'use_calibration': use_calibration
            },
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print("=" * 100)
        print(f"Final Results: {correct}/{total} = {accuracy:.2f}%")
        print(f"Total NFE: {total_nfe} (avg: {total_nfe/total:.1f})" if total > 0 else "N/A")
        print(f"Total time: {format_time(total_time)}")
        print(f"Avg time/sample: {total_time/total:.2f}s" if total > 0 else "N/A")
        print(f"Saved to: {output_file}")
        print("=" * 100)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="HDO-DLM Evaluation on Math Datasets")

    # Model and dataset
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--verifier_gpu", type=int, default=1, help="GPU for Qwen PRM")
    parser.add_argument("--dataset_path", default="dataset/test500.jsonl")
    parser.add_argument("--output_file", default="hdo_results.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_samples", type=int, default=None)

    # Generation parameters
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)

    # HDO-DLM parameters
    parser.add_argument("--beta", type=float, default=1.0, help="Reward temperature")
    parser.add_argument("--candidate_branching", type=int, default=4, help="Number of candidate children (B)")
    parser.add_argument("--backup_width", type=int, default=4, help="Rollout children per backup (M)")
    parser.add_argument("--max_backup_depth", type=int, default=4, help="Maximum Bellman iterations (D_max)")
    parser.add_argument("--residual_threshold", type=float, default=0.1, help="Early stopping threshold")
    parser.add_argument("--exploration_temp", type=float, default=0.0, help="Edge selection temperature")
    parser.add_argument("--no_calibration", action="store_true", help="Disable Bellman calibration (ablation)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed generation info")

    args = parser.parse_args()

    evaluator = HDOEvaluator(
        model_name=args.model_name,
        verifier_gpu=args.verifier_gpu,
        device=args.device,
        beta=args.beta,
        candidate_branching=args.candidate_branching,
        backup_width=args.backup_width,
        max_backup_depth=args.max_backup_depth,
        residual_threshold=args.residual_threshold,
        exploration_temp=args.exploration_temp,
    )

    evaluator.load_dataset(args.dataset_path)

    evaluator.evaluate(
        output_file=args.output_file,
        max_samples=args.max_samples,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        use_calibration=not args.no_calibration,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
