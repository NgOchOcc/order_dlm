"""
Verifier wrapper for Qwen2.5-Math-PRM-7B
Provides reward/value estimation for HDO-DLM
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


class Qwen25MathPRM:
    """Qwen2.5-Math Process Reward Model Verifier"""

    def __init__(self, model_name="Qwen/Qwen2.5-Math-PRM-7B", device="auto"):
        """
        Initialize Qwen2.5-Math PRM verifier

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("auto", "cuda", or "cpu")
        """
        print(f"Loading verifier: {model_name}")
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device).eval()

        print(f"Verifier loaded on {self.device}")

    @torch.no_grad()
    def score(self, text, context=""):
        """
        Score a text completion given optional context

        Args:
            text: Generated text to score
            context: Optional context/prompt

        Returns:
            float: Reward score (higher is better)
        """
        if context:
            full_text = context + text
        else:
            full_text = text

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        # Get logits from the model
        outputs = self.model(**inputs, output_hidden_states=True)

        # For PRM, we typically extract a score from the final hidden state
        # or use a specific scoring head if available
        # This is a simplified version - adjust based on actual PRM architecture
        hidden_states = outputs.hidden_states[-1]  # Last layer

        # Take mean of last hidden state as quality score
        # In practice, PRMs may have a specific value head
        score = hidden_states.mean(dim=(1, 2)).item()

        return score

    @torch.no_grad()
    def score_batch(self, texts, contexts=None):
        """
        Score a batch of text completions

        Args:
            texts: List of generated texts
            contexts: Optional list of contexts (same length as texts)

        Returns:
            torch.Tensor: Batch of reward scores
        """
        if contexts is None:
            contexts = [""] * len(texts)

        full_texts = [c + t for c, t in zip(contexts, texts)]

        # Tokenize batch with padding
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Get outputs
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        # Average pooling over sequence length, accounting for padding
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        scores = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        scores = scores.mean(dim=-1)  # Average over hidden dim

        return scores


class LightweightVerifier:
    """
    Lightweight verifier for quick prototyping
    Uses simple heuristics for math problem verification
    """

    def __init__(self):
        """Initialize lightweight verifier"""
        self.device = "cpu"
        print("Using lightweight verifier (heuristic-based)")

    def score(self, text, context=""):
        """
        Simple heuristic scoring based on:
        - Presence of mathematical symbols
        - Presence of \\boxed{} answer
        - Text length (longer solutions may be more detailed)

        Args:
            text: Generated text
            context: Optional context

        Returns:
            float: Heuristic score
        """
        score = 0.0

        # Bonus for boxed answer
        if "\\boxed{" in text:
            score += 2.0

        # Bonus for mathematical symbols
        math_symbols = ['=', '+', '-', '*', '/', '^', '\\', 'frac', 'sqrt']
        symbol_count = sum(1 for sym in math_symbols if sym in text)
        score += min(symbol_count * 0.2, 2.0)

        # Length bonus (up to 500 chars)
        score += min(len(text) / 500.0, 1.0)

        # Penalty for very short solutions
        if len(text) < 20:
            score -= 1.0

        return score

    def score_batch(self, texts, contexts=None):
        """Batch scoring using heuristics"""
        scores = torch.tensor([self.score(t, c if contexts else "")
                               for t, (c, t) in enumerate(zip(contexts or [""] * len(texts), texts))])
        return scores


def get_verifier(verifier_type="qwen", **kwargs):
    """
    Factory function to get appropriate verifier

    Args:
        verifier_type: "qwen" for Qwen2.5-Math-PRM or "lightweight" for heuristic
        **kwargs: Additional arguments passed to verifier constructor

    Returns:
        Verifier instance
    """
    if verifier_type == "qwen":
        return Qwen25MathPRM(**kwargs)
    elif verifier_type == "lightweight":
        return LightweightVerifier()
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")
