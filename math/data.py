"""
Dataset utilities for math PPO fine-tuning.

These helpers reproduce the prompt preparation logic from `RHRL_PPO.ipynb`.
They fetch Hendrycks MATH problems, optionally filter by subject, and expose
train/eval splits with the `query` column required by TRL/PPO trainers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


@dataclass
class MathDatasetConfig:
    """Configuration for building PPO prompt datasets."""

    hf_dataset_id: str = "nlile/hendrycks-MATH-benchmark"
    hf_split: str = "test"
    subject: Optional[str] = "algebra"
    sample_limit: Optional[int] = 256
    test_fraction: float = 0.2
    seed: int = 42


def _ensure_query_column(ds: Dataset) -> Dataset:
    if "prompt" in ds.column_names and "query" not in ds.column_names:
        ds = ds.add_column("query", ds["prompt"])
    return ds


def _to_final_record(example: dict) -> dict:
    return {
        "prompt": example["problem"],
        "final_answer": example["solution"],
        "level": example.get("level"),
        "subject": example.get("subject"),
    }


def build_math_datasets(config: MathDatasetConfig) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and preprocess math prompts for PPO training/evaluation.

    Returns:
        train_ds: dataset used for PPO rollouts (always provided)
        eval_ds: optional evaluation split (None if `test_fraction` is falsy)
    """

    ds = load_dataset(config.hf_dataset_id, split=config.hf_split)

    if config.subject:
        target = config.subject.lower()
        ds = ds.filter(lambda ex: (ex.get("subject") or "").lower() == target)

    if config.sample_limit:
        sample_count = min(config.sample_limit, len(ds))
        ds = ds.shuffle(seed=config.seed).select(range(sample_count))

    ds = ds.map(_to_final_record, remove_columns=ds.column_names)
    keep_cols = ["prompt", "final_answer", "level", "subject"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    if config.test_fraction and len(ds) > 1:
        split = ds.train_test_split(test_size=config.test_fraction, seed=config.seed)
        train_ds = _ensure_query_column(split["train"])
        eval_ds = _ensure_query_column(split["test"])
        return train_ds, eval_ds

    train_ds = _ensure_query_column(ds)
    return train_ds, None

