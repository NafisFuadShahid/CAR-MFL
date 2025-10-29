"""
Federated Averaging (FedAvg) Implementation for Multimodal Client-Server Architecture
Supports multiple modalities, weighted averaging, and extensible aggregation strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import OrderedDict
import copy
import logging
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for federated aggregation"""
    aggregation_method: str = "weighted"  # weighted, uniform, adaptive
    min_clients: int = 2
    clip_grad_norm: Optional[float] = None
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    momentum: float = 0.0  # Server-side momentum
    adaptive_learning_rate: bool = False


class FedAvg:
    """
    Federated Averaging aggregator for multimodal federated learning.

    Features:
    - Weighted averaging based on client data size
    - Support for multiple modalities (text, image, audio, etc.)
    - Gradient clipping and differential privacy
    - Server-side momentum
    - Adaptive aggregation strategies
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Initialize FedAvg aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self.global_model_state = None
        self.momentum_buffer = None
        self.round_number = 0

    def aggregate(
            self,
            client_models: List[OrderedDict],
            client_weights: Optional[List[float]] = None,
            client_info: Optional[List[Dict[str, Any]]] = None
    ) -> OrderedDict:
        """
        Aggregate client models using FedAvg algorithm.

        Args:
            client_models: List of client model state dicts
            client_weights: List of client weights (e.g., data sizes)
            client_info: Additional client information (modalities, metrics, etc.)

        Returns:
            Aggregated global model state dict
        """
        if len(client_models) < self.config.min_clients:
            raise ValueError(f"Need at least {self.config.min_clients} clients, got {len(client_models)}")

        # Normalize weights
        if client_weights is None:
            client_weights = [1.0] * len(client_models)

        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        logger.info(f"Round {self.round_number}: Aggregating {len(client_models)} client models")
        logger.info(f"Client weights: {[f'{w:.3f}' for w in normalized_weights]}")

        # Aggregate based on method
        if self.config.aggregation_method == "weighted":
            global_state = self._weighted_average(client_models, normalized_weights)
        elif self.config.aggregation_method == "uniform":
            global_state = self._uniform_average(client_models)
        elif self.config.aggregation_method == "adaptive":
            global_state = self._adaptive_average(client_models, normalized_weights, client_info)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")

        # Apply gradient clipping if enabled
        if self.config.clip_grad_norm is not None:
            global_state = self._clip_gradients(global_state, self.config.clip_grad_norm)

        # Apply differential privacy if enabled
        if self.config.differential_privacy:
            global_state = self._add_dp_noise(global_state)

        # Apply server-side momentum if enabled
        if self.config.momentum > 0:
            global_state = self._apply_momentum(global_state)

        self.global_model_state = global_state
        self.round_number += 1

        return global_state

    def _weighted_average(
            self,
            client_models: List[OrderedDict],
            weights: List[float]
    ) -> OrderedDict:
        """Perform weighted averaging of client models."""
        global_state = OrderedDict()

        # Get keys from first model
        keys = client_models[0].keys()

        for key in keys:
            # Stack all client parameters for this key
            client_params = [model[key].float() for model in client_models]

            # Weighted average
            global_state[key] = sum(
                w * param for w, param in zip(weights, client_params)
            )

        return global_state

    def _uniform_average(self, client_models: List[OrderedDict]) -> OrderedDict:
        """Perform uniform (unweighted) averaging of client models."""
        weights = [1.0 / len(client_models)] * len(client_models)
        return self._weighted_average(client_models, weights)

    def _adaptive_average(
            self,
            client_models: List[OrderedDict],
            weights: List[float],
            client_info: Optional[List[Dict[str, Any]]]
    ) -> OrderedDict:
        """
        Perform adaptive averaging based on client performance metrics.

        Adjusts weights based on client loss, accuracy, or other metrics.
        """
        if client_info is None:
            logger.warning("No client info provided for adaptive averaging, falling back to weighted")
            return self._weighted_average(client_models, weights)

        # Extract performance metrics (e.g., validation loss)
        adaptive_weights = []
        for info, base_weight in zip(client_info, weights):
            # Lower loss = higher weight
            loss = info.get('val_loss', 1.0)
            adaptive_weight = base_weight / (loss + 1e-6)
            adaptive_weights.append(adaptive_weight)

        # Normalize
        total = sum(adaptive_weights)
        adaptive_weights = [w / total for w in adaptive_weights]

        logger.info(f"Adaptive weights: {[f'{w:.3f}' for w in adaptive_weights]}")

        return self._weighted_average(client_models, adaptive_weights)

    def _clip_gradients(self, state_dict: OrderedDict, max_norm: float) -> OrderedDict:
        """Apply gradient clipping to model parameters."""
        clipped_state = OrderedDict()

        for key, param in state_dict.items():
            norm = torch.norm(param)
            if norm > max_norm:
                clipped_state[key] = param * (max_norm / (norm + 1e-6))
            else:
                clipped_state[key] = param

        return clipped_state

    def _add_dp_noise(self, state_dict: OrderedDict) -> OrderedDict:
        """Add differential privacy noise to parameters."""
        noisy_state = OrderedDict()

        # Calculate noise scale based on DP parameters
        sensitivity = 1.0  # Assuming normalized updates
        noise_scale = sensitivity / self.config.dp_epsilon

        for key, param in state_dict.items():
            noise = torch.randn_like(param) * noise_scale
            noisy_state[key] = param + noise

        logger.info(f"Added DP noise with ε={self.config.dp_epsilon}, δ={self.config.dp_delta}")

        return noisy_state

    def _apply_momentum(self, state_dict: OrderedDict) -> OrderedDict:
        """Apply server-side momentum to aggregated updates."""
        if self.momentum_buffer is None:
            self.momentum_buffer = copy.deepcopy(state_dict)
            return state_dict

        momentum_state = OrderedDict()

        for key in state_dict.keys():
            # Update momentum buffer
            self.momentum_buffer[key] = (
                    self.config.momentum * self.momentum_buffer[key] +
                    (1 - self.config.momentum) * state_dict[key]
            )
            momentum_state[key] = self.momentum_buffer[key]

        return momentum_state

    def aggregate_modality_specific(
            self,
            client_models: List[OrderedDict],
            modality_weights: Dict[str, List[float]],
            modality_prefix: Dict[str, List[str]]
    ) -> OrderedDict:
        """
        Aggregate models with modality-specific weighting.

        Args:
            client_models: List of client model state dicts
            modality_weights: Dict mapping modality to list of client weights
            modality_prefix: Dict mapping modality to list of layer name prefixes

        Returns:
            Aggregated global model state dict
        """
        global_state = OrderedDict()

        # Get all keys from first model
        all_keys = client_models[0].keys()

        for key in all_keys:
            # Determine which modality this parameter belongs to
            modality = self._identify_modality(key, modality_prefix)

            if modality and modality in modality_weights:
                # Use modality-specific weights
                weights = modality_weights[modality]
                weights = [w / sum(weights) for w in weights]

                client_params = [model[key].float() for model in client_models]
                global_state[key] = sum(
                    w * param for w, param in zip(weights, client_params)
                )
            else:
                # Use uniform weighting for shared layers
                client_params = [model[key].float() for model in client_models]
                global_state[key] = sum(client_params) / len(client_params)

        return global_state

    def _identify_modality(
            self,
            key: str,
            modality_prefix: Dict[str, List[str]]
    ) -> Optional[str]:
        """Identify which modality a parameter belongs to based on its name."""
        for modality, prefixes in modality_prefix.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                return modality
        return None

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Return statistics about the aggregation process."""
        return {
            'round': self.round_number,
            'config': self.config.__dict__,
            'has_momentum': self.momentum_buffer is not None
        }

    def reset(self):
        """Reset aggregator state."""
        self.global_model_state = None
        self.momentum_buffer = None
        self.round_number = 0
        logger.info("FedAvg aggregator reset")


class SecureAggregation:
    """
    Secure aggregation wrapper for privacy-preserving federated learning.
    Implements secure multi-party computation protocols.
    """

    def __init__(self, num_clients: int, threshold: int):
        """
        Initialize secure aggregation.

        Args:
            num_clients: Total number of clients
            threshold: Minimum number of clients needed for reconstruction
        """
        self.num_clients = num_clients
        self.threshold = threshold

    def generate_masks(self, model_shape: OrderedDict) -> List[OrderedDict]:
        """Generate secret masks for each client."""
        masks = []

        for _ in range(self.num_clients):
            mask = OrderedDict()
            for key, param in model_shape.items():
                mask[key] = torch.randn_like(param)
            masks.append(mask)

        return masks

    def aggregate_with_masks(
            self,
            masked_updates: List[OrderedDict],
            masks: List[OrderedDict]
    ) -> OrderedDict:
        """Aggregate masked updates and cancel out masks."""
        # Sum all masked updates
        aggregated = OrderedDict()
        keys = masked_updates[0].keys()

        for key in keys:
            aggregated[key] = sum(update[key] for update in masked_updates)

            # Cancel out masks
            mask_sum = sum(mask[key] for mask in masks)
            aggregated[key] = aggregated[key] - mask_sum

        return aggregated


def compute_client_weights(client_data_sizes: List[int]) -> List[float]:
    """
    Compute client weights based on dataset sizes.

    Args:
        client_data_sizes: List of dataset sizes for each client

    Returns:
        List of normalized weights
    """
    total_size = sum(client_data_sizes)
    return [size / total_size for size in client_data_sizes]


def compute_model_divergence(
        model1: OrderedDict,
        model2: OrderedDict,
        metric: str = "l2"
) -> float:
    """
    Compute divergence between two models.

    Args:
        model1: First model state dict
        model2: Second model state dict
        metric: Distance metric (l2, cosine)

    Returns:
        Divergence score
    """
    divergence = 0.0

    for key in model1.keys():
        if key in model2:
            if metric == "l2":
                divergence += torch.norm(model1[key] - model2[key]).item() ** 2
            elif metric == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(
                    model1[key].flatten(),
                    model2[key].flatten(),
                    dim=0
                )
                divergence += (1 - cos_sim).item()

    return divergence


if __name__ == "__main__":
    # Example usage
    print("FedAvg Multimodal Implementation")
    print("=" * 50)

    # Create dummy client models
    dummy_model = OrderedDict({
        'text_encoder.weight': torch.randn(100, 50),
        'image_encoder.weight': torch.randn(100, 50),
        'fusion.weight': torch.randn(50, 100),
        'classifier.weight': torch.randn(10, 50)
    })

    client_models = [
        OrderedDict({k: v + torch.randn_like(v) * 0.1 for k, v in dummy_model.items()})
        for _ in range(5)
    ]

    # Simulate different client data sizes
    client_data_sizes = [100, 150, 200, 120, 180]
    client_weights = compute_client_weights(client_data_sizes)

    # Create aggregator
    config = AggregationConfig(
        aggregation_method="weighted",
        clip_grad_norm=1.0,
        momentum=0.9
    )

    fedavg = FedAvg(config)

    # Aggregate models
    global_model = fedavg.aggregate(
        client_models=client_models,
        client_weights=client_weights
    )

    print(f"\nAggregation completed!")
    print(f"Global model keys: {list(global_model.keys())}")
    print(f"Stats: {fedavg.get_aggregation_stats()}")