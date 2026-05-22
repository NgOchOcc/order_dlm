"""Verifier for HDO-DLM"""

import torch
import re


class LightweightVerifier:
    """Simple heuristic verifier"""

    def __init__(self):
        self.device = "cpu"

    def score(self, text, context=""):
        score = 0.0
        if "\\boxed{" in text:
            score += 2.0
        math_symbols = ['=', '+', '-', '*', '/', '^', '\\', 'frac', 'sqrt']
        symbol_count = sum(1 for sym in math_symbols if sym in text)
        score += min(symbol_count * 0.2, 2.0)
        score += min(len(text) / 500.0, 1.0)
        if len(text) < 20:
            score -= 1.0
        return score

    def score_batch(self, texts, contexts=None):
        scores = [self.score(t) for t in texts]
        return torch.tensor(scores)


class QwenPRM_vLLM:
    """Qwen2.5-Math-PRM using vLLM"""

    def __init__(self, gpu_id=1):
        """Initialize Qwen PRM with vLLM on specified GPU"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Install vllm: pip install vllm")

        print(f"Loading Qwen PRM with vLLM on GPU {gpu_id}...")
        self.llm = LLM(
            model="Qwen/Qwen2.5-Math-PRM-7B",
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=1
        )
        print(f"✓ Qwen PRM loaded on GPU {gpu_id}")

    def score(self, text, context=""):
        full_text = context + text if context else text
        try:
            outputs = self.llm.generate([full_text], self.sampling_params)
            # Use average logprob as score
            if outputs[0].outputs[0].logprobs:
                logprobs = [lp for token_logprobs in outputs[0].outputs[0].logprobs
                           for lp in token_logprobs.values()]
                score = sum(logprobs) / len(logprobs) if logprobs else 0.0
            else:
                score = 0.0
            return float(score)
        except:
            return 0.0

    def score_batch(self, texts, contexts=None):
        if contexts is None:
            contexts = [""] * len(texts)
        full_texts = [c + t for c, t in zip(contexts, texts)]

        try:
            outputs = self.llm.generate(full_texts, self.sampling_params)
            scores = []
            for output in outputs:
                if output.outputs[0].logprobs:
                    logprobs = [lp for token_logprobs in output.outputs[0].logprobs
                               for lp in token_logprobs.values()]
                    score = sum(logprobs) / len(logprobs) if logprobs else 0.0
                else:
                    score = 0.0
                scores.append(score)
            return torch.tensor(scores)
        except:
            return torch.zeros(len(texts))


def get_verifier(verifier_type="lightweight", gpu_id=1):
    """Get verifier instance"""
    if verifier_type == "lightweight":
        return LightweightVerifier()
    elif verifier_type == "qwen_vllm":
        return QwenPRM_vLLM(gpu_id=gpu_id)
    else:
        raise ValueError(f"Unknown verifier: {verifier_type}")
