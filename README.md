# PPO from Scratch

A minimal, configurable implementation of **Proximal Policy Optimisation** (PPO) in PyTorch, compatible with Gymnasium environments.  

---

## Features

- **Vectorised training & evaluation** with Gymnasium’s `SyncVectorEnv`/`AsyncVectorEnv`  
- **Generalised Advantage Estimation** (GAE) with optional normalisation  
- **Surrogate‐ratio & value‐function clipping** for stable updates  
- **Entropy bonus** (scaled by action‐dim) to encourage exploration  
- **Reproducible seeding** via per‐env RNGs  
- **TensorBoard logging** out of the box  
- **GIF‐recording** of policy rollouts

---

## Example rollout

<figure>
  <img src="docs/policy_eval.gif" alt="PPO policy on LunarLander"/>
  <figcaption><em>Policy evaluation on LunarLander-v3 environment with strong wind.</em></figcaption>
</figure>

## Installation

Install the package in a virtual environment, e.g. using `venv`:

  ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
  ```

## Configuration

By default, all hyperparameters are defined in [`default.yaml`](configs/default.yaml).  
This file controls training settings, model architecture, and the environment to be used.

You can modify this configuration or provide your own YAML file and pass it to the trainer.

## Quickstart

```python
from trainer.train import PPOTrainer

# 1) Initialise trainer (loads default.yaml)
trainer = PPOTrainer()

# 2) Train
trainer.train()

# 3) Evaluate average return
mean_r = trainer.mean_reward_eval()
print(f"Mean episode return: {mean_r:.2f}")

# 4) Record a GIF
trainer.render_policy_eval_gif()
```

---

## Math RLHF Pipeline

The repository now includes a math-specific PPO flow that reuses the math SFT checkpoint and reward model from the `RHRL_PPO` notebook.

1. Install the optional dependencies:
   ```bash
   pip install -e .[math]
   ```
2. Update [`configs/math_default.yaml`](configs/math_default.yaml) with paths to your SFT policy (`math.policy.policy_checkpoint`) and reward model (`math.reward_model.reward_checkpoint`).
3. Launch training:
   ```bash
   python trainer/math_train.py --config configs/math_default.yaml
   ```

During training the script logs PPO losses and reward components (reward-model score, format, correctness, brevity) so you can monitor alignment. Checkpoints and JSON summaries are written to the directory specified by `training.save_dir`. Use the optional evaluation section to score the fine-tuned policy on a held-out set of math prompts.
