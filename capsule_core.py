"""
Capsule Network Core Module.

This module implements the core components of a Capsule Network as described in
"Dynamic Routing Between Capsules" (Sabour et al., 2017), with extensions for
several baseline routing strategies.

Architecture Overview:
    Input (28×28) → Conv1 (9×9, stride=1) → 20×20×256
                  → PrimaryCaps (9×9, stride=2) → 6×6×32×8 = 1152 capsules
                  → DigitCaps (routing) → 10×16
                  → Decoder → 28×28 reconstruction

Routing Strategies:
    - Dynamic: Standard dynamic routing by agreement
    - Uniform: Equal routing weights (baseline)
    - Random: Random routing weights (baseline)
    - TopK: Route only to top-k most similar capsules
    - Sparse: Threshold-based sparse routing

Example:
    >>> model = CapsuleNetwork(in_channels=1, num_classes=10, routing_strategy='dynamic')
    >>> v, c_ij, b_ij, history = model(images)
    >>> reconstruction = model.reconstruct(v, labels)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def squash(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """
    Apply the squashing non-linearity to capsule vectors.

    The squash function ensures that short vectors get shrunk to almost zero
    length and long vectors get shrunk to a length slightly below 1.

    Formula:
        v = (||s||² / (1 + ||s||²)) * (s / ||s||)

    Args:
        x: Input tensor of capsule vectors.
        dim: Dimension along which to compute the norm.
        eps: Small constant for numerical stability.

    Returns:
        Squashed tensor with the same shape as input, where vector lengths
        represent existence probabilities (in range [0, 1)).
    """
    squared_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)




class PrimaryCapsules(nn.Module):
    """
    Primary Capsules layer that converts convolutional features to capsule vectors.

    This layer applies a convolutional operation and reshapes the output into
    a grid of capsule vectors, each representing a specific type of visual entity
    at a particular location.

    Architecture:
        Input: [batch, in_channels, H, W]
        Conv2d: kernel=9×9, stride=2 → [batch, num_capsules * capsule_dim, H', W']
        Reshape + Squash → [batch, H' * W' * num_capsules, capsule_dim]

    Attributes:
        num_capsules: Number of capsule types (default: 32).
        capsule_dim: Dimension of each capsule vector (default: 8).
        conv: Convolutional layer for feature extraction.
    """

    def __init__(
        self,
        in_channels: int,
        num_capsules: int = 32,
        capsule_dim: int = 8,
    ) -> None:
        """
        Initialize the Primary Capsules layer.

        Args:
            in_channels: Number of input channels from previous conv layer.
            num_capsules: Number of capsule types to create.
            capsule_dim: Dimension of each capsule's output vector.
        """
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=9,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Primary Capsules layer.

        Args:
            x: Input tensor of shape [batch, in_channels, H, W].

        Returns:
            Capsule output tensor of shape [batch, num_total_capsules, capsule_dim],
            where num_total_capsules = H' * W' * num_capsules.
        """
        x = self.conv(x)
        batch_size, _, height, width = x.size()

        # Reshape: [B, C*D, H, W] → [B, C, D, H, W]
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, height, width)

        # Reorder dimensions and flatten spatial grid: [B, H*W*C, D]
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(batch_size, height * width * self.num_capsules, self.capsule_dim)

        return squash(x)


class RoutingStrategy(nn.Module):
    """
    Abstract base class for capsule routing strategies.

    This class defines the interface for all routing mechanisms that determine
    how lower-level capsules send their outputs to higher-level capsules.

    Attributes:
        num_input_caps: Number of input (primary) capsules.
        input_dim: Dimension of input capsule vectors.
        num_output_caps: Number of output (digit) capsules.
        output_dim: Dimension of output capsule vectors.
        routing_iters: Number of routing iterations.
        temperature: Temperature parameter for softmax (controls sharpness).
        W: Transformation weight matrix [1, O, I, D_out, D_in].
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize the routing strategy.

        Args:
            num_input_caps: Number of input capsules.
            input_dim: Dimension of each input capsule.
            num_output_caps: Number of output capsules.
            output_dim: Dimension of each output capsule.
            routing_iters: Number of routing iterations.
            temperature: Temperature for softmax normalization.
        """
        super().__init__()
        self.num_input_caps = num_input_caps
        self.input_dim = input_dim
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim
        self.routing_iters = routing_iters
        self.temperature = temperature

        # Initialize transformation weights with small random values
        weight = torch.randn(1, num_output_caps, num_input_caps, output_dim, input_dim) * 0.01
        self.W = nn.Parameter(weight)

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Forward pass through the routing mechanism.

        Args:
            u: Input capsule vectors [batch, num_input_caps, input_dim].

        Returns:
            Tuple containing:
                - v: Output capsule vectors [batch, num_output_caps, output_dim]
                - c: Final coupling coefficients [batch, num_output_caps, num_input_caps]
                - b: Final agreement scores [batch, num_output_caps, num_input_caps]
                - routing_history: List of routing state dictionaries per iteration
        """
        raise NotImplementedError("Subclasses must implement forward()")


class DynamicRouting(RoutingStrategy):
    """
    Dynamic Routing by Agreement.

    Implements the original dynamic routing algorithm from Sabour et al. (2017).
    Capsules iteratively refine their coupling coefficients based on agreement
    between predictions and actual outputs.

    Algorithm:
        1. Compute predictions u_hat = W @ u for all input-output pairs
        2. Initialize logits b = 0
        3. For each iteration:
           a. Compute coupling coefficients c = softmax(b/T) over outputs
           b. Compute weighted sum s = Σ(c * u_hat)
           c. Squash to get output v = squash(s)
           d. Update logits b += u_hat · v (agreement)
    """

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Apply dynamic routing by agreement.

        Args:
            u: Input capsule vectors [batch, num_input_caps, input_dim].

        Returns:
            Tuple of (output capsules, coupling coefficients, agreement scores, history).
        """
        batch_size = u.size(0)

        # Transform input capsules: [B, 1, I, D, 1] × [1, O, I, D_out, D_in] → [B, O, I, D_out]
        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        # Initialize routing logits to zero
        b = u.new_zeros(batch_size, self.num_output_caps, self.num_input_caps)

        routing_history: List[Dict[str, Any]] = []

        for iteration in range(self.routing_iters):
            # Compute coupling coefficients via softmax over output capsules
            c = F.softmax(b / self.temperature, dim=1)

            # Weighted sum and squash
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)
            v = squash(s)

            # Record routing state for analysis
            routing_history.append({
                "iteration": iteration,
                "coupling_coeffs": c.clone(),
                "agreement_scores": b.clone(),
                "digit_capsules": v.clone(),
            })

            # Update agreement scores (except on last iteration)
            if iteration < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)

        return v, c, b, routing_history


class UniformRouting(RoutingStrategy):
    """
    Uniform Routing (Baseline).

    Routes all primary capsules equally to all digit capsules.
    Serves as a baseline to demonstrate the importance of dynamic routing.

    Each coupling coefficient is set to 1/num_output_caps, giving equal
    weight to all output capsules regardless of agreement.
    """

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Apply uniform routing (equal weights to all outputs).

        Args:
            u: Input capsule vectors [batch, num_input_caps, input_dim].

        Returns:
            Tuple of (output capsules, coupling coefficients, agreement scores, history).
        """
        batch_size = u.size(0)

        # Transform input capsules
        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        # Uniform coupling: each output capsule gets equal weight
        c = torch.ones(
            batch_size, self.num_output_caps, self.num_input_caps, device=u.device
        ) / self.num_output_caps

        # Weighted sum and squash
        c_expanded = c.unsqueeze(-1).expand_as(u_hat)
        s = (c_expanded * u_hat).sum(dim=2)
        v = squash(s)

        routing_history = [
            {
                "iteration": 0,
                "coupling_coeffs": c.clone(),
                "agreement_scores": c.clone(),
                "digit_capsules": v.clone(),
            }
        ]

        return v, c, c, routing_history


class RandomRouting(RoutingStrategy):
    """
    Random Routing (Baseline).

    Assigns random coupling coefficients, normalized via softmax.
    Serves as a baseline to show that learned routing is essential.
    """

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Apply random routing (random weights normalized by softmax).

        Args:
            u: Input capsule vectors [batch, num_input_caps, input_dim].

        Returns:
            Tuple of (output capsules, coupling coefficients, agreement scores, history).
        """
        batch_size = u.size(0)

        # Transform input capsules
        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        # Random coupling coefficients, normalized via softmax
        random_logits = torch.rand_like(u_hat[:, :, :, 0])
        c = F.softmax(random_logits, dim=1)

        # Weighted sum and squash
        c_expanded = c.unsqueeze(-1).expand_as(u_hat)
        s = (c_expanded * u_hat).sum(dim=2)
        v = squash(s)

        routing_history = [
            {
                "iteration": 0,
                "coupling_coeffs": c.clone(),
                "agreement_scores": c.clone(),
                "digit_capsules": v.clone(),
            }
        ]

        return v, c, c, routing_history


class TopKRouting(RoutingStrategy):
    """
    Top-K Routing Strategy.

    Routes each primary capsule only to its top-k most similar output capsules.
    Similarity is measured by the L2 norm of the prediction vectors.

    This sparse routing approach can improve efficiency and interpretability
    by focusing each capsule's contribution on the most relevant outputs.

    Attributes:
        k: Number of top output capsules to route to (computed from k_ratio).
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
        temperature: float = 1.0,
        k_ratio: float = 0.25,
    ) -> None:
        """
        Initialize Top-K Routing.

        Args:
            num_input_caps: Number of input capsules.
            input_dim: Dimension of each input capsule.
            num_output_caps: Number of output capsules.
            output_dim: Dimension of each output capsule.
            routing_iters: Number of routing iterations (unused, kept for interface).
            temperature: Temperature parameter (unused, kept for interface).
            k_ratio: Fraction of output capsules to route to (default: 0.25).
        """
        super().__init__(
            num_input_caps, input_dim, num_output_caps, output_dim,
            routing_iters, temperature
        )
        self.k = max(1, int(num_output_caps * k_ratio))

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Apply top-k routing based on prediction similarity.

        Args:
            u: Input capsule vectors [batch, num_input_caps, input_dim].

        Returns:
            Tuple of (output capsules, coupling coefficients, similarity scores, history).
        """
        batch_size = u.size(0)

        # Transform input capsules
        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        # Compute similarity as L2 norm of predictions
        similarities = torch.norm(u_hat, dim=-1)  # [B, O, I]

        # Select top-k outputs for each input capsule
        _, topk_indices = torch.topk(similarities, self.k, dim=1)  # [B, k, I]

        # Create sparse coupling coefficients
        c = torch.zeros_like(similarities)
        batch_indices = torch.arange(batch_size, device=u.device).unsqueeze(1).unsqueeze(2)
        input_indices = torch.arange(self.num_input_caps, device=u.device).unsqueeze(0).unsqueeze(0)
        c[batch_indices, topk_indices, input_indices] = 1.0 / self.k

        # Weighted sum and squash
        c_expanded = c.unsqueeze(-1).expand_as(u_hat)
        s = (c_expanded * u_hat).sum(dim=2)
        v = squash(s)

        routing_history = [
            {
                "iteration": 0,
                "coupling_coeffs": c.clone(),
                "agreement_scores": similarities.clone(),
                "digit_capsules": v.clone(),
            }
        ]

        return v, c, similarities, routing_history


class SparseRouting(RoutingStrategy):
    """
    Sparse Routing Strategy.

    Applies a threshold to filter out weak connections, creating sparse
    coupling coefficients. Connections below the threshold quantile are zeroed.

    This approach reduces noise from irrelevant capsules and can improve
    both performance and interpretability.

    Attributes:
        sparsity_threshold: Quantile threshold for filtering (default: 0.7).
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
        temperature: float = 1.0,
        sparsity_threshold: float = 0.7,
    ) -> None:
        """
        Initialize Sparse Routing.

        Args:
            num_input_caps: Number of input capsules.
            input_dim: Dimension of each input capsule.
            num_output_caps: Number of output capsules.
            output_dim: Dimension of each output capsule.
            routing_iters: Number of routing iterations (unused).
            temperature: Temperature parameter (unused).
            sparsity_threshold: Quantile threshold (0-1) for filtering connections.
        """
        super().__init__(
            num_input_caps, input_dim, num_output_caps, output_dim,
            routing_iters, temperature
        )
        self.sparsity_threshold = sparsity_threshold

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Apply sparse routing with threshold-based filtering.

        Args:
            u: Input capsule vectors [batch, num_input_caps, input_dim].

        Returns:
            Tuple of (output capsules, coupling coefficients, similarity scores, history).
        """
        batch_size = u.size(0)

        # Transform input capsules
        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        # Compute similarity scores
        similarities = torch.norm(u_hat, dim=-1)

        # Apply threshold filtering
        threshold = torch.quantile(similarities.flatten(), self.sparsity_threshold)
        c = (similarities > threshold).float()

        # Normalize: each primary capsule's coefficients sum to 1
        c_sum = c.sum(dim=1, keepdim=True)
        # For capsules with no connections above threshold, use softmax fallback
        c = torch.where(c_sum > 0, c / c_sum, F.softmax(similarities, dim=1))

        # Weighted sum and squash
        c_expanded = c.unsqueeze(-1).expand_as(u_hat)
        s = (c_expanded * u_hat).sum(dim=2)
        v = squash(s)

        routing_history = [
            {
                "iteration": 0,
                "coupling_coeffs": c.clone(),
                "agreement_scores": similarities.clone(),
                "digit_capsules": v.clone(),
            }
        ]

        return v, c, similarities, routing_history


class DigitCapsules(nn.Module):
    """
    Digit Capsules layer with configurable routing strategy.

    This layer wraps the routing mechanism and provides a unified interface
    for different routing strategies. It manages the transformation from
    primary capsules to digit (class) capsules.

    Attributes:
        num_input_caps: Number of input (primary) capsules.
        input_dim: Dimension of input capsule vectors.
        num_output_caps: Number of output (digit) capsules.
        output_dim: Dimension of output capsule vectors.
        routing_iters: Number of routing iterations.
        routing_strategy: Name of the routing strategy being used.
        routing_module: The actual routing module instance.
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
        routing_strategy: str = "dynamic",
        **routing_kwargs: Any,
    ) -> None:
        """
        Initialize the Digit Capsules layer.

        Args:
            num_input_caps: Number of input capsules.
            input_dim: Dimension of each input capsule.
            num_output_caps: Number of output capsules (typically num_classes).
            output_dim: Dimension of each output capsule.
            routing_iters: Number of routing iterations.
            routing_strategy: One of 'dynamic', 'uniform', 'random', 'top_k',
                            or 'sparse'.
            **routing_kwargs: Additional arguments for specific routing strategies
                            (e.g., k_ratio for top_k).
        """
        super().__init__()
        self.num_input_caps = num_input_caps
        self.input_dim = input_dim
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim
        self.routing_iters = routing_iters
        self.routing_strategy = routing_strategy
        self.routing_module = self._create_routing_module(routing_strategy, **routing_kwargs)

    def _create_routing_module(
        self, strategy: str, **kwargs: Any
    ) -> RoutingStrategy:
        """
        Factory method to create the appropriate routing module.

        Args:
            strategy: Name of the routing strategy.
            **kwargs: Strategy-specific parameters.

        Returns:
            Configured routing module instance.

        Raises:
            ValueError: If strategy name is not recognized.
        """
        base_args = (
            self.num_input_caps,
            self.input_dim,
            self.num_output_caps,
            self.output_dim,
            self.routing_iters,
        )

        if strategy == "dynamic":
            return DynamicRouting(*base_args)
        if strategy == "uniform":
            return UniformRouting(*base_args)
        if strategy == "random":
            return RandomRouting(*base_args)
        if strategy == "top_k":
            return TopKRouting(*base_args, k_ratio=kwargs.get("k_ratio", 0.25))
        if strategy == "sparse":
            return SparseRouting(
                *base_args, sparsity_threshold=kwargs.get("sparsity_threshold", 0.7)
            )

        raise ValueError(f"Unknown routing strategy: {strategy}")

    def forward(
        self, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Forward pass through the digit capsules layer.

        Args:
            u: Input from primary capsules [batch, num_input_caps, input_dim].

        Returns:
            Tuple from the routing module: (v, c, b, routing_history).
        """
        return self.routing_module(u)


class CapsuleNetwork(nn.Module):
    """
    Complete Capsule Network for image classification.

    Implements the full CapsNet architecture with configurable routing strategy,
    including the reconstruction decoder for regularization.

    Architecture:
        Conv1: 256 filters, 9×9, stride 1 → ReLU
        Primary Capsules: 32 types × 8D vectors on 6×6 grid (for 28×28 input)
        Digit Capsules: 10 capsules × 16D vectors (one per class)
        Decoder: 3-layer MLP for reconstruction (160 → 512 → 1024 → H×W×C)

    The length of each digit capsule vector represents the probability that
    the corresponding class is present in the input.

    Attributes:
        img_width: Input image width.
        img_height: Input image height.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        conv1: First convolutional layer.
        primary: Primary capsules layer.
        digits: Digit capsules layer with routing.
        decoder: Reconstruction decoder network.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        routing_iters: int = 3,
        routing_strategy: str = "dynamic",
        img_width: int = 28,
        img_height: int = 28,
        num_primary_capsules: int = 32,
        primary_capsule_dim: int = 8,
        **routing_kwargs: Any,
    ) -> None:
        """
        Initialize the Capsule Network.

        Args:
            in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
            num_classes: Number of output classes.
            routing_iters: Number of dynamic routing iterations.
            routing_strategy: Routing algorithm to use.
            img_width: Input image width in pixels.
            img_height: Input image height in pixels.
            num_primary_capsules: Number of primary capsule types.
            primary_capsule_dim: Dimension of primary capsule vectors.
            **routing_kwargs: Additional arguments for routing strategy.
        """
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.in_channels = in_channels

        # First convolutional layer: extracts basic features
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=9, stride=1)

        # Primary capsules: convert conv features to capsule vectors
        self.primary = PrimaryCapsules(
            256, num_capsules=num_primary_capsules, capsule_dim=primary_capsule_dim
        )

        # Calculate spatial dimensions after conv layers
        # Conv1: (W - 9) / 1 + 1 = W - 8
        # Primary: (W - 8 - 9) / 2 + 1 = (W - 17) / 2 + 1
        width_after_conv1 = (img_width - 9) // 1 + 1
        height_after_conv1 = (img_height - 9) // 1 + 1
        width_after_primary = (width_after_conv1 - 9) // 2 + 1
        height_after_primary = (height_after_conv1 - 9) // 2 + 1

        num_primary_caps_total = num_primary_capsules * width_after_primary * height_after_primary

        # Digit capsules with configurable routing
        self.digits = DigitCapsules(
            num_input_caps=num_primary_caps_total,
            input_dim=primary_capsule_dim,
            num_output_caps=num_classes,
            output_dim=16,
            routing_iters=routing_iters,
            routing_strategy=routing_strategy,
            **routing_kwargs,
        )

        self.num_classes = num_classes

        # Reconstruction decoder for regularization
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, in_channels * img_width * img_height),
            nn.Sigmoid(),
        )

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """
        Forward pass through the capsule network.

        Args:
            x: Input images [batch, channels, height, width].

        Returns:
            Tuple containing:
                - v: Digit capsule vectors [batch, num_classes, 16]
                - c_ij: Coupling coefficients from routing
                - b_ij: Agreement scores from routing
                - routing_history: List of routing states per iteration
        """
        # Feature extraction
        x = F.relu(self.conv1(x))

        # Primary capsules
        u = self.primary(x)

        # Digit capsules with routing
        v, c_ij, b_ij, routing_history = self.digits(u)

        return v, c_ij, b_ij, routing_history

    def reconstruct(
        self, v: Tensor, labels: Optional[Tensor] = None
    ) -> Tensor:
        """
        Reconstruct input image from digit capsule vectors.

        Uses the decoder network to reconstruct the input from the capsule
        representation. If labels are provided, uses the corresponding capsule;
        otherwise uses the capsule with the longest vector (predicted class).

        Args:
            v: Digit capsule vectors [batch, num_classes, 16].
            labels: Optional ground truth labels for masking [batch].
                   If None, uses predicted labels (argmax of lengths).

        Returns:
            Reconstructed image pixels [batch, H * W * C] (flattened).
        """
        lengths = torch.norm(v, dim=-1)

        if labels is None:
            labels = lengths.argmax(dim=1)

        # Create one-hot mask for the target class
        mask = F.one_hot(labels, num_classes=self.num_classes).float()

        # Mask and flatten: keep only the target capsule's vector
        masked = (v * mask.unsqueeze(-1)).reshape(v.size(0), -1)

        # Decode to reconstruction
        reconstruction = self.decoder(masked)

        return reconstruction


def capsule_loss(y_true: Tensor, v: Tensor) -> Tensor:
    """
    Compute the margin loss for capsule networks.

    The margin loss penalizes:
    - False negatives: When a class is present but its capsule is short
    - False positives: When a class is absent but its capsule is long

    Formula:
        L_k = T_k * max(0, m+ - ||v_k||)² + λ * (1 - T_k) * max(0, ||v_k|| - m-)²

    Args:
        y_true: One-hot encoded ground truth labels [batch, num_classes].
        v: Digit capsule vectors [batch, num_classes, capsule_dim].

    Returns:
        Scalar loss value (mean over batch).
    """
    m_plus = 0.9   # Upper margin for present classes
    m_minus = 0.1  # Lower margin for absent classes
    lambda_weight = 0.5  # Down-weighting factor for absent classes

    lengths = torch.norm(v, dim=-1)

    # Loss for present classes: penalize if length < m+
    present_loss = y_true * F.relu(m_plus - lengths) ** 2

    # Loss for absent classes: penalize if length > m-
    absent_loss = lambda_weight * (1.0 - y_true) * F.relu(lengths - m_minus) ** 2

    # Sum over classes, mean over batch
    return (present_loss + absent_loss).sum(dim=1).mean()


def reconstruction_loss(images: Tensor, reconstructions: Tensor) -> Tensor:
    """
    Compute the reconstruction loss (MSE) for regularization.

    The reconstruction loss encourages the capsule network to preserve
    all information about the input needed for reconstruction.

    Args:
        images: Original input images [batch, C, H, W].
        reconstructions: Reconstructed images [batch, C * H * W] (flattened).

    Returns:
        Mean squared error between original and reconstructed images.
    """
    images_flat = images.view(images.size(0), -1)
    return F.mse_loss(reconstructions, images_flat, reduction="mean")
