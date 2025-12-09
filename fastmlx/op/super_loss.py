"""SuperLoss - Self-paced curriculum learning loss wrapper."""

from __future__ import annotations

import math
from typing import Any, Callable, List, MutableMapping, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from .op import LossOp, Op


def _lambert_w(x: np.ndarray, n_iterations: int = 10) -> np.ndarray:
    """Compute Lambert W function using Halley's method.

    The Lambert W function is defined as W(x) where W(x) * exp(W(x)) = x.
    Used in SuperLoss for computing confidence weights.

    Args:
        x: Input values (should be >= -1/e).
        n_iterations: Number of iterations for Newton-Halley method.

    Returns:
        Lambert W function values.
    """
    # Initial guess
    w = np.log(1 + x + 1e-8)

    # Halley's method iterations
    for _ in range(n_iterations):
        ew = np.exp(w)
        wew = w * ew
        wew_minus_x = wew - x

        # Halley's iteration
        w_next = w - wew_minus_x / (ew * (w + 1) - (w + 2) * wew_minus_x / (2 * w + 2 + 1e-8) + 1e-8)

        # Handle numerical issues
        w = np.where(np.isfinite(w_next), w_next, w)

    return w


class SuperLoss(LossOp):
    """Self-paced curriculum learning loss wrapper.

    SuperLoss automatically down-weights samples with high loss (potentially noisy
    or hard samples) during training, implementing a form of curriculum learning.
    Particularly useful for training with noisy labels.

    The loss is computed as:
        super_loss = (loss - tau) * sigma + lambda * (log(sigma))^2

    where sigma is the confidence weight learned per-sample and tau is the
    running average of the base loss.

    Args:
        loss_op: The base loss operation to wrap.
        inputs: Keys for the inputs (should match loss_op's inputs).
        outputs: Key for the output loss.
        lam: Regularization parameter controlling confidence penalty.
        tau_mode: How to compute tau. 'exp_avg' for exponential moving average,
                  'constant' for a fixed value.
        tau_init: Initial value for tau (average loss estimate).
        momentum: Momentum for exponential moving average of tau.
        confidence_output: Optional key to store confidence weights.
        mode: Mode(s) in which to run. Defaults to None (all modes).

    Reference:
        Castells et al., "SuperLoss: A Generic Loss for Robust Curriculum Learning",
        NeurIPS 2020.

    Example:
        >>> base_loss = CrossEntropy(inputs=("y_pred", "y"), outputs="ce")
        >>> super_loss = SuperLoss(
        ...     loss_op=base_loss,
        ...     inputs=("y_pred", "y"),
        ...     outputs="super_ce",
        ...     lam=1.0
        ... )
    """

    def __init__(
        self,
        loss_op: LossOp,
        inputs: Sequence[str],
        outputs: str,
        lam: float = 1.0,
        tau_mode: str = "exp_avg",
        tau_init: float = 0.0,
        momentum: float = 0.9,
        confidence_output: Optional[str] = None,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs, outputs, mode)
        self.loss_op = loss_op
        self.lam = lam
        self.tau_mode = tau_mode
        self.tau_init = tau_init
        self.momentum = momentum
        self.confidence_output = confidence_output

        # State for tracking tau
        self._tau: float = tau_init
        self._initialized: bool = False

    def _compute_sigma(self, loss: np.ndarray, tau: float) -> np.ndarray:
        """Compute optimal confidence weight sigma using Lambert W function.

        sigma = exp(-W((loss - tau) / lambda))

        where W is the Lambert W function.
        """
        # Compute argument for Lambert W
        z = (loss - tau) / (self.lam + 1e-8)

        # Compute Lambert W
        w = _lambert_w(z)

        # Compute sigma
        sigma = np.exp(-w)

        # Clamp to valid range
        sigma = np.clip(sigma, 1e-8, 1.0)

        return sigma

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        # Compute base loss (per-sample, not reduced)
        y_pred, y_true = data

        # Get per-sample loss from base loss op
        # We need unreduced loss for SuperLoss
        base_loss = self.loss_op.forward(data, state)

        # Convert to numpy for Lambert W computation
        if isinstance(base_loss, mx.array):
            loss_np = np.array(base_loss)
        else:
            loss_np = np.array([base_loss])

        # Handle scalar loss (already reduced)
        if loss_np.ndim == 0:
            loss_np = np.array([float(loss_np)])

        # Update tau (running average of loss)
        current_loss_mean = float(np.mean(loss_np))

        if not self._initialized:
            self._tau = current_loss_mean if self.tau_init == 0 else self.tau_init
            self._initialized = True
        elif self.tau_mode == "exp_avg":
            self._tau = self.momentum * self._tau + (1 - self.momentum) * current_loss_mean

        # Compute confidence weights
        sigma = self._compute_sigma(loss_np, self._tau)

        # Compute SuperLoss
        # super_loss = (loss - tau) * sigma + lambda * (log(sigma))^2
        log_sigma = np.log(sigma + 1e-8)
        super_loss = (loss_np - self._tau) * sigma + self.lam * (log_sigma ** 2)

        # Return mean super loss
        mean_super_loss = mx.array(np.mean(super_loss))

        # Optionally return confidence for logging
        if self.confidence_output is not None:
            state[self.confidence_output] = mx.array(sigma)

        return mean_super_loss


class ConfidenceWeightedLoss(LossOp):
    """Apply sample-wise confidence weighting to any loss.

    A simpler alternative to SuperLoss that uses a fixed confidence schedule
    based on loss values.

    Args:
        loss_op: The base loss operation to wrap.
        inputs: Keys for the inputs.
        outputs: Key for the output loss.
        threshold: Loss threshold above which samples are down-weighted.
        min_weight: Minimum weight for high-loss samples.
        mode: Mode(s) in which to run.

    Example:
        >>> ConfidenceWeightedLoss(
        ...     loss_op=CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        ...     inputs=("y_pred", "y"),
        ...     outputs="weighted_ce",
        ...     threshold=2.0
        ... )
    """

    def __init__(
        self,
        loss_op: LossOp,
        inputs: Sequence[str],
        outputs: str,
        threshold: float = 2.0,
        min_weight: float = 0.1,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs, outputs, mode)
        self.loss_op = loss_op
        self.threshold = threshold
        self.min_weight = min_weight

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        # Compute base loss
        base_loss = self.loss_op.forward(data, state)

        # Convert to numpy
        if isinstance(base_loss, mx.array):
            loss_np = np.array(base_loss)
        else:
            loss_np = np.array([base_loss])

        if loss_np.ndim == 0:
            loss_np = np.array([float(loss_np)])

        # Compute confidence weights based on threshold
        weights = np.ones_like(loss_np)
        high_loss_mask = loss_np > self.threshold

        # Linear decay for high loss samples
        if np.any(high_loss_mask):
            excess = loss_np[high_loss_mask] - self.threshold
            decay = np.exp(-excess)
            weights[high_loss_mask] = np.maximum(decay, self.min_weight)

        # Weighted mean
        weighted_loss = np.sum(weights * loss_np) / (np.sum(weights) + 1e-8)

        return mx.array(weighted_loss)


class GradientWeightedLoss(LossOp):
    """Weight samples by gradient magnitude for curriculum learning.

    Down-weights samples with extremely large gradients, which often
    correspond to outliers or label noise.

    Args:
        loss_op: The base loss operation to wrap.
        inputs: Keys for the inputs.
        outputs: Key for the output loss.
        percentile: Percentile for gradient clipping threshold.
        mode: Mode(s) in which to run.

    Example:
        >>> GradientWeightedLoss(
        ...     loss_op=CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        ...     inputs=("y_pred", "y"),
        ...     outputs="gw_ce"
        ... )
    """

    def __init__(
        self,
        loss_op: LossOp,
        inputs: Sequence[str],
        outputs: str,
        percentile: float = 95.0,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs, outputs, mode)
        self.loss_op = loss_op
        self.percentile = percentile
        self._gradient_history: List[float] = []
        self._threshold: Optional[float] = None

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        # Compute base loss
        base_loss = self.loss_op.forward(data, state)

        # For gradient weighting, we estimate sample difficulty by loss magnitude
        if isinstance(base_loss, mx.array):
            loss_val = float(base_loss.item()) if base_loss.ndim == 0 else float(np.mean(np.array(base_loss)))
        else:
            loss_val = float(base_loss)

        # Track loss history
        self._gradient_history.append(loss_val)
        if len(self._gradient_history) > 1000:
            self._gradient_history = self._gradient_history[-1000:]

        # Update threshold
        if len(self._gradient_history) >= 10:
            self._threshold = np.percentile(self._gradient_history, self.percentile)

        # Apply weighting if we have a threshold
        if self._threshold is not None and loss_val > self._threshold:
            weight = self._threshold / (loss_val + 1e-8)
            weight = max(weight, 0.1)  # Minimum weight
            return base_loss * weight

        return base_loss
