"""
HDO-DLM: Harmonic Doob Ordering for Diffusion Language Models
Main algorithm implementation
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, List

from harmonic_estimator import HarmonicEstimator


class HDODLM:
    """
    HDO-DLM: Harmonic Doob Ordering for Diffusion Language Models

    Implements test-time scaling via Doob h-transform on the mask lattice.
    Key principle: denoise the edge with largest harmonic future-reward advantage,
    not the token with highest confidence.
    """

    def __init__(
        self,
        model,
        tokenizer,
        verifier,
        mask_id=126336,
        beta=1.0,
        num_particles=1,
        candidate_branching=4,
        backup_width=4,
        max_backup_depth=4,
        residual_threshold=0.1,
        exploration_temp=0.0,
        alpha_smooth=0.5,
        epsilon_mixture=0.0,
    ):
        """
        Initialize HDO-DLM sampler

        Args:
            model: Base DLM model
            tokenizer: Tokenizer
            verifier: Reward model/verifier
            mask_id: Token ID for mask
            beta: Reward temperature
            num_particles: Number of particles (N)
            candidate_branching: Number of candidate children per state (B)
            backup_width: Number of rollout children per backup (M)
            max_backup_depth: Maximum Bellman iterations (D_max)
            residual_threshold: Threshold for early stopping (eps)
            exploration_temp: Temperature for Doob edge selection (tau)
            alpha_smooth: Smoothing parameter for Bellman updates
            epsilon_mixture: Epsilon for epsilon-greedy mixture with base kernel
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.mask_id = mask_id
        self.beta = beta

        # Algorithm hyperparameters
        self.num_particles = num_particles
        self.candidate_branching = candidate_branching
        self.backup_width = backup_width
        self.max_backup_depth = max_backup_depth
        self.residual_threshold = residual_threshold
        self.exploration_temp = exploration_temp
        self.alpha_smooth = alpha_smooth
        self.epsilon_mixture = epsilon_mixture

        # Initialize harmonic estimator
        self.harmonic_est = HarmonicEstimator(
            model, tokenizer, verifier, mask_id, beta
        )

        self.device = model.device

    @torch.no_grad()
    def sample_candidate_children(
        self,
        x_t: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        B: int,
        strategy: str = "mixed",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample B candidate child states from current state x_t

        Strategies:
        - "confidence": Top-confidence positions
        - "entropy": Top-entropy (uncertain) positions
        - "random": Random positions
        - "mixed": Combination of above

        Args:
            x_t: Current state (batch_size, seq_len)
            attention_mask: Optional attention mask
            B: Number of candidates
            strategy: Candidate selection strategy

        Returns:
            candidates: Candidate child states (batch_size, B, seq_len)
            positions: Selected positions for each candidate (batch_size, B)
            tokens: Selected tokens for each candidate (batch_size, B)
        """
        batch_size, seq_len = x_t.shape
        mask_index = (x_t == self.mask_id)

        # Get model logits
        logits = self.model(x_t, attention_mask=attention_mask).logits  # (batch_size, seq_len, vocab_size)

        # Compute confidence and entropy per position
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # (batch_size, seq_len)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (batch_size, seq_len)

        # Mask out non-masked positions
        confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf).to(self.device))
        entropy = torch.where(mask_index, entropy, torch.tensor(-np.inf).to(self.device))

        candidates_list = []
        positions_list = []
        tokens_list = []

        for b in range(batch_size):
            if strategy == "confidence":
                # Top confidence positions
                _, top_positions = torch.topk(confidence[b], k=min(B, mask_index[b].sum().item()))
            elif strategy == "entropy":
                # Top entropy positions
                _, top_positions = torch.topk(entropy[b], k=min(B, mask_index[b].sum().item()))
            elif strategy == "random":
                # Random masked positions
                masked_positions = torch.where(mask_index[b])[0]
                perm = torch.randperm(len(masked_positions))[:B]
                top_positions = masked_positions[perm]
            elif strategy == "mixed":
                # Mix of strategies
                num_masked = mask_index[b].sum().item()
                k_conf = max(1, B // 3)
                k_ent = max(1, B // 3)
                k_rand = B - k_conf - k_ent

                _, conf_pos = torch.topk(confidence[b], k=min(k_conf, num_masked))
                _, ent_pos = torch.topk(entropy[b], k=min(k_ent, num_masked))

                masked_positions = torch.where(mask_index[b])[0]
                rand_pos = masked_positions[torch.randperm(len(masked_positions))[:k_rand]]

                top_positions = torch.cat([conf_pos, ent_pos, rand_pos])[:B]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # For each position, sample or take argmax token
            batch_candidates = []
            batch_tokens = []

            for pos in top_positions:
                # Sample token from distribution at this position
                if self.exploration_temp > 0:
                    token = torch.multinomial(
                        F.softmax(logits[b, pos] / self.exploration_temp, dim=0),
                        num_samples=1
                    ).item()
                else:
                    token = torch.argmax(logits[b, pos]).item()

                # Create child state
                child = x_t[b].clone()
                child[pos] = token

                batch_candidates.append(child)
                batch_tokens.append(token)

            # Stack candidates
            candidates_batch = torch.stack(batch_candidates)  # (B, seq_len)
            candidates_list.append(candidates_batch)
            positions_list.append(top_positions)
            tokens_list.append(torch.tensor(batch_tokens).to(self.device))

        candidates = torch.stack(candidates_list)  # (batch_size, B, seq_len)
        positions = torch.stack(positions_list)  # (batch_size, B)
        tokens = torch.stack(tokens_list)  # (batch_size, B)

        return candidates, positions, tokens

    @torch.no_grad()
    def create_transition_kernel(self, x_parent, attention_mask=None, num_to_reveal=1):
        """
        Create a transition kernel K_t(· | x_parent) that samples child states

        Args:
            x_parent: Parent state
            attention_mask: Optional attention mask
            num_to_reveal: Number of tokens to reveal per transition

        Returns:
            Function that samples child state
        """
        def kernel(x_t, attn_mask):
            # Sample candidate and return one
            candidates, _, _ = self.sample_candidate_children(
                x_t.unsqueeze(0) if x_t.dim() == 1 else x_t,
                attn_mask,
                B=1,
                strategy="confidence"
            )
            return candidates[:, 0, :]  # Return first candidate

        return kernel

    @torch.no_grad()
    def compute_doob_scores(
        self,
        x_t: torch.Tensor,
        candidates: torch.Tensor,
        positions: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        use_calibration: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Doob edge scores for candidate children

        score_b = log K_t(y_b | x_t) + ell_{t-1}(y_b)

        Args:
            x_t: Current state (batch_size, seq_len)
            candidates: Candidate children (batch_size, B, seq_len)
            positions: Positions for each candidate (batch_size, B)
            tokens: Tokens for each candidate (batch_size, B)
            attention_mask: Optional attention mask
            use_calibration: Whether to use Bellman calibration

        Returns:
            scores: Doob scores (batch_size, B)
            info: Dictionary with detailed information
        """
        batch_size, B, seq_len = candidates.shape

        # Flatten candidates for batch processing
        candidates_flat = candidates.view(-1, seq_len)  # (batch_size * B, seq_len)

        # Compute log K_t term (base transition probability)
        logits = self.model(x_t, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)

        log_K_t = []
        for b in range(batch_size):
            for j in range(B):
                pos = positions[b, j].item()
                tok = tokens[b, j].item()
                log_K_t.append(log_probs[b, pos, tok].item())

        log_K_t = torch.tensor(log_K_t).to(self.device).view(batch_size, B)

        # Compute h estimates for candidates
        if use_calibration:
            # Use Bellman-calibrated h estimate
            kernel = self.create_transition_kernel(x_t, attention_mask)

            ell_candidates = []
            depths_used = []
            residuals = []

            for i in range(batch_size * B):
                cand = candidates_flat[i:i+1]

                # Adaptive depth
                depth = self.harmonic_est.adaptive_depth_schedule(
                    cand, base_depth=2, max_depth=self.max_backup_depth
                )

                # Calibrated estimate
                ell, depth_used, residual = self.harmonic_est.calibrated_h_estimate(
                    cand,
                    kernel,
                    attention_mask,
                    max_depth=depth,
                    M=self.backup_width,
                    alpha=self.alpha_smooth,
                    eps=self.residual_threshold,
                )

                ell_candidates.append(ell[0])
                depths_used.append(depth_used)
                residuals.append(residual[0].item())

            ell_candidates = torch.tensor(ell_candidates).to(self.device).view(batch_size, B)
            avg_depth = np.mean(depths_used)
            avg_residual = np.mean(residuals)

        else:
            # Use only initial clean prediction estimate
            ell_candidates = self.harmonic_est.initial_h_estimate(
                candidates_flat, attention_mask
            ).view(batch_size, B)
            avg_depth = 0
            avg_residual = 0.0

        # Compute Doob scores
        doob_scores = log_K_t + ell_candidates

        info = {
            "avg_backup_depth": avg_depth,
            "avg_residual": avg_residual,
            "log_K_t": log_K_t,
            "ell_h": ell_candidates,
        }

        return doob_scores, info

    @torch.no_grad()
    def select_doob_edge(
        self,
        scores: torch.Tensor,
        candidates: torch.Tensor,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select next child state based on Doob scores

        Args:
            scores: Doob scores (batch_size, B)
            candidates: Candidate children (batch_size, B, seq_len)
            temperature: Sampling temperature (0 = greedy)

        Returns:
            selected: Selected child states (batch_size, seq_len)
            indices: Selected indices (batch_size,)
        """
        batch_size, B, seq_len = candidates.shape

        if temperature <= 0:
            # Greedy selection
            indices = torch.argmax(scores, dim=1)
        else:
            # Softmax sampling
            probs = F.softmax(scores / temperature, dim=1)
            indices = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Gather selected candidates
        selected = candidates[torch.arange(batch_size), indices]

        return selected, indices

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        steps: int = 128,
        gen_length: int = 256,
        block_length: int = 32,
        use_calibration: bool = True,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate completions using HDO-DLM

        Args:
            prompt: Input prompt (batch_size, prompt_len)
            attention_mask: Optional attention mask
            steps: Number of denoising steps
            gen_length: Length of generation
            block_length: Block size for block-level denoising
            use_calibration: Whether to use Bellman calibration
            verbose: Print detailed information

        Returns:
            output: Generated sequences (batch_size, prompt_len + gen_length)
            stats: Dictionary with generation statistics
        """
        batch_size = prompt.shape[0]

        # Initialize with masked tokens
        x = torch.full(
            (batch_size, prompt.shape[1] + gen_length),
            self.mask_id,
            dtype=torch.long
        ).to(self.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=self.device)
            ], dim=-1)

        # Block-level generation
        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks

        stats = {
            "total_nfe": 0,
            "avg_depth_per_step": [],
            "avg_residual_per_step": [],
        }

        for block_idx in range(num_blocks):
            block_start = prompt.shape[1] + block_idx * block_length
            block_end = block_start + block_length

            if verbose:
                print(f"\nBlock {block_idx+1}/{num_blocks}: positions [{block_start}:{block_end}]")

            # Get number of tokens to reveal per step in this block
            block_mask_index = (x[:, block_start:block_end] == self.mask_id)
            num_tokens_per_step = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for step in range(steps_per_block):
                # Sample candidate children
                candidates, positions, tokens = self.sample_candidate_children(
                    x, attention_mask, B=self.candidate_branching, strategy="mixed"
                )

                stats["total_nfe"] += 1  # Forward pass for candidate sampling

                # Compute Doob scores
                doob_scores, info = self.compute_doob_scores(
                    x, candidates, positions, tokens, attention_mask, use_calibration
                )

                stats["total_nfe"] += self.candidate_branching  # h estimation
                if use_calibration:
                    stats["total_nfe"] += info["avg_backup_depth"] * self.backup_width

                stats["avg_depth_per_step"].append(info["avg_backup_depth"])
                stats["avg_residual_per_step"].append(info["avg_residual"])

                # Select next state via Doob edge selection
                x_next, selected_idx = self.select_doob_edge(
                    doob_scores, candidates, temperature=self.exploration_temp
                )

                # Epsilon-mixture with base kernel
                if self.epsilon_mixture > 0 and np.random.rand() < self.epsilon_mixture:
                    # Fall back to confidence-based selection
                    logits = self.model(x, attention_mask=attention_mask).logits
                    probs = F.softmax(logits, dim=-1)
                    confidence = probs.max(dim=-1).values
                    mask_index = (x == self.mask_id)
                    confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf).to(self.device))

                    _, top_pos = torch.topk(confidence[0], k=1)
                    x_next = x.clone()
                    x_next[0, top_pos] = torch.argmax(logits[0, top_pos], dim=-1)

                x = x_next

                if verbose and step % 10 == 0:
                    print(f"  Step {step}/{steps_per_block} | "
                          f"Avg Depth: {info['avg_backup_depth']:.1f} | "
                          f"Avg Residual: {info['avg_residual']:.3f}")

        stats["avg_depth"] = np.mean(stats["avg_depth_per_step"]) if stats["avg_depth_per_step"] else 0
        stats["avg_residual"] = np.mean(stats["avg_residual_per_step"]) if stats["avg_residual_per_step"] else 0

        return x, stats

    def _get_num_transfer_tokens(self, mask_index, steps):
        """
        Compute number of tokens to reveal per step (from eval_llada.py)

        Args:
            mask_index: Boolean mask of masked positions (batch_size, block_len)
            steps: Number of steps

        Returns:
            num_transfer_tokens: Tokens per step (batch_size, steps)
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        ) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens
