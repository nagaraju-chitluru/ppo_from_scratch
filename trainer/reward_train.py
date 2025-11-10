import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import RewardTrainer


@dataclass
class RewardDatasetConfig:
    hf_dataset_id: str = "kira/math-dpo"
    hf_split: str = "train"
    sample_limit: Optional[int] = 100
    seed: int = 42


@dataclass
class RewardPolicyConfig:
    policy_checkpoint: str = "/content/models/unsloth_sft_model"
    fallback_model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    device_map: Optional[str | dict] = "auto"
    torch_dtype: Optional[str] = "float16"
    load_in_8bit: bool = True
    use_fast_tokenizer: bool = False


@dataclass
class RewardTrainingConfig:
    output_dir: str = "/content/drive/MyDrive/rl/reward_model"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-5
    num_train_epochs: int = 3
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 1
    report_to: str = "none"
    fp16: bool = False
    seed: int = 42
    max_length: int = 512


@dataclass
class RewardLoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name or name == "auto":
        return None
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


def prepare_dataset(cfg: RewardDatasetConfig) -> Dataset:
    ds = load_dataset(cfg.hf_dataset_id, split=cfg.hf_split)
    if cfg.sample_limit:
        limit = min(len(ds), cfg.sample_limit)
        ds = ds.shuffle(seed=cfg.seed).select(range(limit))
    keep_cols = [c for c in ["prompt", "chosen", "rejected"] if c in ds.column_names]
    if len(keep_cols) < 3:
        raise ValueError(
            f"Dataset must contain prompt/chosen/rejected. Columns: {ds.column_names}"
        )
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    return ds


def load_policy(cfg: RewardPolicyConfig):
    path = Path(cfg.policy_checkpoint).expanduser().resolve()
    model_id = str(path) if path.exists() else cfg.fallback_model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=cfg.use_fast_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(cfg.torch_dtype)

    quant_config = None
    if cfg.load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        device_map=cfg.device_map,
        torch_dtype=dtype,
        quantization_config=quant_config,
    )

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model, lora_cfg: RewardLoraConfig):
    config = LoraConfig(
        task_type="SEQ_CLS",
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
    )
    return get_peft_model(model, config)


def tokenize_preferences(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def preprocess(batch):
        prompts = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        chosen_enc = tokenizer(
            [p + "\n" + c for p, c in zip(prompts, chosen)],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        rejected_enc = tokenizer(
            [p + "\n" + r for p, r in zip(prompts, rejected)],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    processed = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return processed


def run_training(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)

    dataset_cfg = RewardDatasetConfig(**cfg.reward.dataset)
    policy_cfg = RewardPolicyConfig(**cfg.reward.policy)
    training_cfg = RewardTrainingConfig(**cfg.reward.training)
    lora_cfg = RewardLoraConfig(**cfg.lora)

    model, tokenizer = load_policy(policy_cfg)
    model = apply_lora(model, lora_cfg)
    model.train()

    raw_dataset = prepare_dataset(dataset_cfg)
    dataset = tokenize_preferences(raw_dataset, tokenizer, training_cfg.max_length)

    training_args = TrainingArguments(
        output_dir=training_cfg.output_dir,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        num_train_epochs=training_cfg.num_train_epochs,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        remove_unused_columns=False,
        report_to=training_cfg.report_to,
        fp16=training_cfg.fp16,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    torch.manual_seed(training_cfg.seed)
    np.random.seed(training_cfg.seed)

    trainer.train()
    trainer.model.save_pretrained(training_cfg.output_dir)
    tokenizer.save_pretrained(training_cfg.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train math reward model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reward_default.yaml"),
        help="Path to reward training config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)
