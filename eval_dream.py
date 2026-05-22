"""Dream Diffusion Model Evaluation on Math Datasets"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import re
import argparse
import os
import time
from datetime import datetime

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


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
        return verify(pred_parsed, gt_parsed)
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


class DreamEvaluator:
    """Dream Model Evaluator"""

    def __init__(self, model_name="Dream-org/Dream-v0-Instruct-7B", device="auto"):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.is_instruct = 'Instruct' in model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager"  # Disable flash attention
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
            return messages
        else:
            # For base model
            return f"Problem: {problem}\n\nSolve the problem step by step and put your final answer in \\boxed{{}}.\n\nSolution:"

    def generate(self, problem, steps=512, max_new_tokens=512, temperature=0.2,
                 top_p=0.95, alg="entropy", alg_temp=0.):
        """Generate answer using Dream diffusion generation"""
        messages = self.create_prompt(problem)

        # Apply chat template for Instruct model
        if self.is_instruct:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True
            )
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
        else:
            # For base model
            encoded = self.tokenizer(messages, return_tensors="pt")
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

        # Dream diffusion generation
        output = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
        )

        # Decode generation
        generated_text = self.tokenizer.decode(
            output.sequences[0][len(input_ids[0]):].tolist(),
            skip_special_tokens=False
        )

        # Remove EOS token
        generated_text = generated_text.split(self.tokenizer.eos_token)[0]

        return generated_text.strip()

    def extract_answer(self, text):
        """Extract answer with priority: boxed > patterns > last number"""
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

    def evaluate(self, output_file="results.json", max_samples=None, steps=512,
                 max_new_tokens=512, temperature=0.2, top_p=0.95, alg="entropy", alg_temp=0.):
        """Evaluate on dataset"""
        results = []
        correct = total = 0
        samples = self.dataset[:max_samples] if max_samples else self.dataset

        start_time = time.time()

        print(f"\nEvaluating {len(samples)} samples")
        print(f"Config: steps={steps}, max_new_tokens={max_new_tokens}, temp={temperature}, "
              f"top_p={top_p}, alg={alg}, alg_temp={alg_temp}")
        print("="*100)

        for idx, sample in enumerate(samples):
            problem = sample.get('problem')
            ground_truth = sample.get('answer', '')
            if not problem:
                continue

            try:
                sample_start = time.time()

                generated = self.generate(
                    problem,
                    steps=steps,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    alg=alg,
                    alg_temp=alg_temp
                )
                predicted = self.extract_answer(generated)
                gt_answer = self.extract_answer(ground_truth) if isinstance(ground_truth, str) else str(ground_truth)
                is_correct = verify_answer(predicted, gt_answer)

                if is_correct:
                    correct += 1
                total += 1

                sample_time = time.time() - sample_start
                total_time = time.time() - start_time
                acc = correct / total * 100

                # Print detailed info for each sample
                status = "✓" if is_correct else "✗"
                print(f"{status} [{idx+1}/{len(samples)}] | Acc: {correct}/{total} = {acc:.2f}% | "
                      f"Time: {format_time(sample_time)} | Total: {format_time(total_time)}")
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
            'config': {
                'steps': steps,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'alg': alg,
                'alg_temp': alg_temp
            },
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
    parser = argparse.ArgumentParser(description="Dream Evaluation on Math Datasets")
    parser.add_argument("--model_name", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--dataset_path", default="dataset/test500.jsonl")
    parser.add_argument("--output_file", default="dream_results.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--alg", default="entropy", choices=["entropy", "confidence"])
    parser.add_argument("--alg_temp", type=float, default=0.0)
    args = parser.parse_args()

    evaluator = DreamEvaluator(model_name=args.model_name, device=args.device)
    evaluator.load_dataset(args.dataset_path)
    evaluator.evaluate(
        output_file=args.output_file,
        max_samples=args.max_samples,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        alg=args.alg,
        alg_temp=args.alg_temp
    )


if __name__ == "__main__":
    main()
