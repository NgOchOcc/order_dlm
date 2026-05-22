"""
Inference and Evaluation Script for LlaDA/Dream Model on Math500 Dataset
Uses math_verify library for accurate mathematical equivalence checking
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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


class Math500Evaluator:
    """Evaluator for Math500 dataset"""

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Base", device: str = "auto"):
        """
        Initialize the evaluator with model and tokenizer

        Args:
            model_name: HuggingFace model name
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()

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

    def create_prompt(self, problem: str, use_cot: bool = True) -> str:
        """
        Create prompt for the model

        Args:
            problem: Math problem text
            use_cot: Whether to use chain-of-thought prompting
        """
        if use_cot:
            prompt = f"""Solve the following math problem step by step.

Problem: {problem}

Solution: Let's solve this step by step.
"""
        else:
            prompt = f"""Solve the following math problem.

Problem: {problem}

Answer:"""

        return prompt

    def generate_answer(self, problem: str, max_new_tokens: int = 512,
                       temperature: float = 0.7, use_cot: bool = True) -> str:
        """
        Generate answer for a given problem

        Args:
            problem: Math problem text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cot: Whether to use chain-of-thought prompting
        """
        prompt = self.create_prompt(problem, use_cot)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove prompt)
        answer = generated_text[len(prompt):].strip()

        return answer

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
        # Remove whitespace
        answer = answer.strip()
        # Remove common formatting
        answer = answer.replace(",", "").replace("$", "").replace("\\", "")
        # Convert to lowercase
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
                # Use math_verify for mathematical equivalence checking
                is_correct = math_equal(predicted, ground_truth)
                return is_correct
            except Exception as e:
                # Fallback to string comparison if math_verify fails
                print(f"math_verify error: {e}, using fallback")
                return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)
        else:
            # Fallback to string comparison
            return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)

    def evaluate_dataset(self, output_file: str = "results.json",
                        use_cot: bool = True, max_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset

        Args:
            output_file: File to save results
            use_cot: Whether to use chain-of-thought prompting
            max_samples: Maximum number of samples to evaluate (None for all)
        """
        results = []
        correct = 0
        total = 0

        samples = self.dataset[:max_samples] if max_samples else self.dataset

        print(f"\nEvaluating on {len(samples)} samples...")

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
                generated = self.generate_answer(problem, use_cot=use_cot)
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
            'use_cot': use_cot,
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
    parser = argparse.ArgumentParser(description="Evaluate LlaDA/Dream model on Math500")
    parser.add_argument(
        "--model_name",
        type=str,
        default="GSAI-ML/LLaDA-8B-Base",
        help="HuggingFace model name"
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
        "--use_cot",
        action="store_true",
        default=True,
        help="Use chain-of-thought prompting"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = Math500Evaluator(
        model_name=args.model_name,
        device=args.device
    )

    # Load dataset
    evaluator.load_math500_dataset(args.dataset_path)

    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        output_file=args.output_file,
        use_cot=args.use_cot,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
