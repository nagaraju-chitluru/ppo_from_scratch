"""
Utilities for math-specific PPO fine-tuning built on top of the base PPO
implementation.  This package bundles helpers to load the policy/value models,
math datasets, and composite reward functions reused across training scripts.
"""

from .data import MathDatasetConfig, build_math_datasets
from .model import (
    MathPolicyConfig,
    load_math_policy,
    MathRewardModelConfig,
    load_math_reward_model,
)
from .reward import MathRewardScorer, MathRewardWeights

__all__ = [
    "MathDatasetConfig",
    "build_math_datasets",
    "MathPolicyConfig",
    "load_math_policy",
    "MathRewardModelConfig",
    "load_math_reward_model",
    "MathRewardScorer",
    "MathRewardWeights",
]

