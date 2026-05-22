"""
Harmonic function estimation for HDO-DLM
Implements backward information function h_t with Bellman calibration
"""

import torch
import numpy as np
import torch.nn.functional as F


class HarmonicEstimator:
    """
    Estimator for the backward information function h_t(x_t) = E[exp(βR(X_0)) | X_t = x_t]
    Uses clean prediction initialization and Bellman calibration
    """

    def __init__(self, model, tokenizer, verifier, mask_id=126336, beta=1.0):
        """
        Initialize harmonic estimator

        Args:
            model: Base DLM model
            tokenizer: Tokenizer for the model
            verifier: Verifier/reward model for scoring completions
            mask_id: Token ID for mask
            beta: Reward temperature (controls reward strength)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.mask_id = mask_id
        self.beta = beta
        self.device = model.device

    @torch.no_grad()
    def clean_predict(self, x_t, attention_mask=None, sample=False, temperature=0.0):
        """
        Generate clean prediction x_0 from partially masked state x_t
        Fill all remaining masks with model's predictions

        Args:
            x_t: Partially denoised sequence (batch_size, seq_len)
            attention_mask: Optional attention mask
            sample: Whether to sample or use argmax
            temperature: Sampling temperature

        Returns:
            x_0: Clean sequence prediction
        """
        x_0 = x_t.clone()
        mask_index = (x_t == self.mask_id)

        if not mask_index.any():
            return x_0

        # Get model predictions
        logits = self.model(x_t, attention_mask=attention_mask).logits

        if sample and temperature > 0:
            # Sample from distribution
            probs = F.softmax(logits / temperature, dim=-1)
            x_0_pred = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(x_t.shape)
        else:
            # Argmax decoding
            x_0_pred = torch.argmax(logits, dim=-1)

        # Fill masked positions
        x_0[mask_index] = x_0_pred[mask_index]

        return x_0

    @torch.no_grad()
    def initial_h_estimate(self, x_t, attention_mask=None, num_samples=1):
        """
        Initial h estimation using clean prediction and verifier

        h^(0)_t(x_t) = exp(β * V(clean_predict(x_t)))

        Args:
            x_t: Partially denoised state (batch_size, seq_len)
            attention_mask: Optional attention mask
            num_samples: Number of clean predictions to average over

        Returns:
            ell_0: Log of h estimate (batch_size,)
        """
        batch_size = x_t.size(0)
        log_h_estimates = []

        for _ in range(num_samples):
            # Generate clean prediction
            x_0_pred = self.clean_predict(x_t, attention_mask, sample=(num_samples > 1))

            # Decode to text for verifier
            texts = [self.tokenizer.decode(x_0_pred[i], skip_special_tokens=True)
                    for i in range(batch_size)]

            # Score with verifier
            if hasattr(self.verifier, 'score_batch'):
                scores = self.verifier.score_batch(texts)
            else:
                scores = torch.tensor([self.verifier.score(t) for t in texts])

            # Apply temperature and log
            log_h = self.beta * scores
            log_h_estimates.append(log_h)

        # Average in log space (logmeanexp)
        log_h_stack = torch.stack(log_h_estimates)  # (num_samples, batch_size)
        ell_0 = torch.logsumexp(log_h_stack, dim=0) - np.log(num_samples)

        return ell_0.to(self.device)

    @torch.no_grad()
    def bellman_backup(self, x_t, ell_t, transition_kernel, attention_mask=None, M=4, num_clean_samples=1):
        """
        Perform one Bellman backup to refine h estimate

        ell_{t-1}(y) = log(1/M * sum_m exp(ell_{t-1}(Y_m)))
        where Y_m ~ K_t(· | x_t)

        Args:
            x_t: Current state (batch_size, seq_len)
            ell_t: Current log-h estimate at x_t (batch_size,)
            transition_kernel: Function to sample children: Y_m ~ K_t(· | x_t)
            attention_mask: Optional attention mask
            M: Number of rollout children per backup
            num_clean_samples: Samples for clean prediction per child

        Returns:
            ell_backup: Bellman backup estimate (batch_size,)
        """
        batch_size = x_t.size(0)
        log_h_children = []

        for _ in range(M):
            # Sample child state from transition kernel
            x_child = transition_kernel(x_t, attention_mask)

            # Get h estimate for child
            ell_child = self.initial_h_estimate(x_child, attention_mask, num_clean_samples)
            log_h_children.append(ell_child)

        # Compute logmeanexp over children
        log_h_stack = torch.stack(log_h_children)  # (M, batch_size)
        ell_backup = torch.logsumexp(log_h_stack, dim=0) - np.log(M)

        return ell_backup

    def bellman_residual(self, ell_t, ell_backup):
        """
        Compute Bellman/martingale residual

        E_t(x_t) = |log h_t(x_t) - log E[h_{t-1}(Y)]|

        where Y ~ K_t(· | x_t)

        Args:
            ell_t: Current log-h estimate (batch_size,)
            ell_backup: Bellman backup estimate (batch_size,)

        Returns:
            residual: Absolute residual (batch_size,)
        """
        residual = torch.abs(ell_t - ell_backup)
        return residual

    @torch.no_grad()
    def calibrated_h_estimate(self, x_t, transition_kernel, attention_mask=None,
                             max_depth=4, M=4, alpha=0.5, eps=0.1, num_clean_samples=1):
        """
        Iteratively refine h estimate using Bellman calibration with adaptive depth

        ell^(d+1)(x) = (1-α) * ell^(0)(x) + α * Bellman_backup(ell^(d))

        Args:
            x_t: Partially denoised state (batch_size, seq_len)
            transition_kernel: Function to sample child states
            attention_mask: Optional attention mask
            max_depth: Maximum number of Bellman iterations
            M: Number of children per backup
            alpha: Smoothing parameter for updates
            eps: Residual threshold for early stopping
            num_clean_samples: Samples for clean prediction

        Returns:
            ell_final: Calibrated log-h estimate (batch_size,)
            depth_used: Actual depth used (int)
            final_residual: Final Bellman residual (batch_size,)
        """
        # Initial estimate from clean prediction
        ell_0 = self.initial_h_estimate(x_t, attention_mask, num_clean_samples)
        ell_current = ell_0.clone()

        depth_used = 0
        final_residual = torch.zeros_like(ell_0)

        for d in range(max_depth):
            # Perform Bellman backup
            ell_backup = self.bellman_backup(
                x_t, ell_current, transition_kernel,
                attention_mask, M, num_clean_samples
            )

            # Compute residual
            residual = self.bellman_residual(ell_current, ell_backup)

            # Update with smoothing
            ell_current = (1 - alpha) * ell_0 + alpha * ell_backup

            depth_used = d + 1
            final_residual = residual

            # Early stopping if residual is small
            if residual.mean() < eps:
                break

        return ell_current, depth_used, final_residual

    def adaptive_depth_schedule(self, x_t, base_depth=2, max_depth=6, alpha_depth=2.0):
        """
        Compute adaptive backup depth based on state properties

        D_t = D_min + D_max * (1 - |S_t| / L)^α

        where |S_t| is number of revealed positions

        Args:
            x_t: Current state (batch_size, seq_len)
            base_depth: Minimum depth
            max_depth: Maximum depth
            alpha_depth: Depth schedule exponent

        Returns:
            depth: Computed depth for this state (int)
        """
        # Count number of non-masked positions
        mask_index = (x_t == self.mask_id)
        num_masked = mask_index.float().sum(dim=1).mean().item()
        total_len = x_t.size(1)

        # Fraction of sequence that is still masked
        masked_fraction = num_masked / total_len

        # More masked → harder to estimate → use deeper backup
        depth = int(base_depth + max_depth * (masked_fraction ** alpha_depth))
        depth = min(depth, max_depth)
        depth = max(depth, base_depth)

        return depth
