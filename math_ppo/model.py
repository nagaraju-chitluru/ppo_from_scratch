"""
Model loading helpers for math PPO fine-tuning.

These functions wrap Hugging Face transformers utilities for:
  * Loading the SFT checkpoint (policy) with a value head
  * Loading the reward model used to score prompt/response pairs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)


@dataclass
class MathPolicyConfig:
    policy_checkpoint: str
    fallback_model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    device_map: Optional[str | dict] = "auto"
    torch_dtype: Optional[str] = None  # "float16", "bfloat16", "float32", etc.
    load_in_8bit: bool = False
    use_fast_tokenizer: bool = False


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name or name == "auto":
        return None
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


def load_math_policy(
    config: MathPolicyConfig,
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the math SFT checkpoint with a value head, plus the reference model.
    Falls back to the public model if the provided checkpoint is missing.
    """
    ckpt_path = None
    model_id = config.fallback_model_id

    if config.policy_checkpoint:
        potential = Path(config.policy_checkpoint).expanduser().resolve()
        if potential.exists():
            ckpt_path = potential
            model_id = str(potential)
        else:
            logger.warning(
                "Policy checkpoint %s not found. Falling back to %s.",
                potential,
                config.fallback_model_id,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=config.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(config.torch_dtype)

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        device_map=config.device_map,
        torch_dtype=dtype,
        load_in_8bit=config.load_in_8bit,
    )
    policy_model.config.pad_token_id = tokenizer.pad_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=config.device_map,
        torch_dtype=dtype,
        load_in_8bit=False,
    )
    ref_model.config.pad_token_id = tokenizer.pad_token_id

    return policy_model, ref_model, tokenizer


@dataclass
class MathRewardModelConfig:
    reward_checkpoint: str
    device_map: Optional[str | dict] = "auto"
    torch_dtype: Optional[str] = "float32"
    load_in_8bit: bool = False
    use_fast_tokenizer: bool = False


def load_math_reward_model(
    config: MathRewardModelConfig,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    ckpt_path = Path(config.reward_checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Reward checkpoint not found at {ckpt_path}. Please provide a valid path."
        )

    dtype = _resolve_dtype(config.torch_dtype)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path,
        device_map=config.device_map,
        torch_dtype=dtype,
        load_in_8bit=config.load_in_8bit,
        num_labels=1,
    )

    reward_tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        use_fast=config.use_fast_tokenizer,
    )

    return reward_model, reward_tokenizer

