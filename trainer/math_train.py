import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from math_ppo import (
    MathDatasetConfig,
    MathPolicyConfig,
    MathRewardModelConfig,
    MathRewardScorer,
    MathRewardWeights,
    build_math_datasets,
    load_math_policy,
    load_math_reward_model,
)

try:
    from trl import PPOConfig, PPOTrainer
except ImportError as exc:  # pragma: no cover - informative error if extras missing
    raise ImportError(
        "trl is required for math PPO training. Install with `pip install .[math]`."
    ) from exc


def _to_device(tensors: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [t.to(device) for t in tensors]


def _prepare_generation_kwargs(
    generation_cfg: Dict[str, Any], tokenizer
) -> Dict[str, Any]:
    kwargs = generation_cfg.copy()
    # Fill EOS/PAD ids if left null in the YAML
    if kwargs.get("eos_token_id") is None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if kwargs.get("pad_token_id") is None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    return kwargs


def _mean_dict(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    agg: Dict[str, List[float]] = {}
    for item in metrics:
        for key, value in item.items():
            agg.setdefault(key, []).append(value)
    return {key: float(np.mean(vals)) for key, vals in agg.items()}


def run_training(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)

    math_cfg = cfg.get("math")
    training_cfg = cfg.get("training")
    generation_cfg = cfg.get("generation") or {}

    if math_cfg is None or training_cfg is None:
        raise ValueError("Config must declare `math` and `training` sections.")

    dataset_cfg = MathDatasetConfig(**math_cfg["dataset"])
    policy_cfg = MathPolicyConfig(**math_cfg["policy"])
    reward_model_cfg = MathRewardModelConfig(**math_cfg["reward_model"])
    reward_weights = MathRewardWeights(**math_cfg.get("reward_weights", {}))

    train_ds, eval_ds = build_math_datasets(dataset_cfg)

    policy_model, ref_model, tokenizer = load_math_policy(policy_cfg)
    reward_model, reward_tokenizer = load_math_reward_model(reward_model_cfg)

    scorer = MathRewardScorer(
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        weights=reward_weights,
        max_tokens_hint=training_cfg.get("max_tokens_hint", 220),
    )

    ppo_config = PPOConfig(
        batch_size=training_cfg["batch_size"],
        mini_batch_size=training_cfg["mini_batch_size"],
        learning_rate=training_cfg["learning_rate"],
        target_kl=training_cfg.get("target_kl"),
        log_with=training_cfg.get("log_with"),
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_ds,
    )

    generation_kwargs = _prepare_generation_kwargs(generation_cfg, tokenizer)
    device = trainer.model.pretrained_model.device
    total_steps = int(training_cfg["total_steps"])

    reward_history: List[Dict[str, float]] = []

    progress = tqdm(range(total_steps), desc="Math PPO steps")
    for step, batch in zip(progress, trainer.dataloader):
        queries: List[str] = batch["query"]
        truths: List[Optional[str]] = batch.get("final_answer", ["" for _ in queries])

        query_tensors = _to_device(
            [
                tokenizer(q, return_tensors="pt").input_ids.squeeze(0)
                for q in queries
            ],
            device,
        )

        response_tensors_full = trainer.generate(
            query_tensors,
            max_new_tokens=training_cfg["max_new_tokens"],
            **generation_kwargs,
        )

        response_tensors = []
        reward_tensors = []
        step_metrics = []
        for query, truth, full_response_ids, query_ids in zip(
            queries, truths, response_tensors_full, query_tensors
        ):
            completion_ids = full_response_ids[len(query_ids) :]
            if completion_ids.numel() == 0:
                completion_ids = torch.tensor(
                    [tokenizer.eos_token_id], device=device, dtype=full_response_ids.dtype
                )
            response_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            rm_stats = scorer.score(query, response_text, truth)

            response_tensors.append(completion_ids)
            reward_tensors.append(torch.tensor(rm_stats["total"], device=device))
            step_metrics.append(rm_stats)

        reward_history.extend(step_metrics)
        stats = trainer.step(query_tensors, response_tensors, reward_tensors)
        trainer.log_stats(stats, batch, reward_tensors)

        averaged = _mean_dict(step_metrics)
        progress.set_postfix({"reward": averaged.get("total", 0.0)})

    save_dir = Path(training_cfg.get("save_dir", "./math_policy"))
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    metrics_path = save_dir / "reward_history.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(reward_history, f, indent=2)

    if eval_ds:
        evaluate(trainer, eval_ds, scorer, tokenizer, training_cfg, save_dir)


def evaluate(
    trainer: PPOTrainer,
    eval_ds,
    scorer: MathRewardScorer,
    tokenizer,
    training_cfg,
    save_dir: Path,
) -> None:
    device = trainer.model.pretrained_model.device
    results = []

    for example in eval_ds:
        prompt = example.get("query") or example.get("prompt")
        truth = example.get("final_answer")
        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).to(
            device
        )
        response_full = trainer.generate(
            [query_tensor],
            max_new_tokens=training_cfg["max_new_tokens"],
        )[0]
        completion_ids = response_full[len(query_tensor) :]
        if completion_ids.numel() == 0:
            completion_ids = torch.tensor(
                [tokenizer.eos_token_id], device=response_full.device, dtype=response_full.dtype
            )
        response_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        rm_stats = scorer.score(prompt, response_text, truth)
        results.append(
            {
                "prompt": prompt,
                "response": response_text,
                "truth": truth,
                **rm_stats,
            }
        )

    eval_path = save_dir / "eval_results.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run math PPO fine-tuning")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/math_default.yaml"),
        help="Path to the math PPO config YAML",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)
