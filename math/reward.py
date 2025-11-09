"""
Composite reward helpers adapted from `RHRL_PPO.ipynb`.

The scorer combines:
* Reward-model logits on prompt/response pairs
* Rule-based checks for boxed answers and brevity
* Optional symbolic equivalence checks via SymPy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except Exception:  # pragma: no cover - sympy optional
    SYMPY_AVAILABLE = False

BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


@dataclass
class MathRewardWeights:
    correct: float = 1.0
    format: float = 0.1
    brevity: float = 0.2
    reward_model: float = 0.2


class MathRewardScorer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        reward_model: torch.nn.Module,
        reward_tokenizer: AutoTokenizer,
        weights: MathRewardWeights = MathRewardWeights(),
        max_tokens_hint: int = 220,
        device: Optional[torch.device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.weights = weights
        self.max_tokens_hint = max_tokens_hint
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.reward_model.to(self.device)
        self.reward_model.eval()

    # -------------------------
    # Reward-model scoring
    # -------------------------
    def score_reward_model(self, prompt: str, response: str) -> float:
        """Compute scalar RM score for a prompt/response pair."""
        text = f"[PROMPT]\n{prompt}\n\n[OUTPUT]\n{response}"
        tokens = self.reward_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_tokens_hint,
        ).to(self.device)

        with torch.no_grad():
            logits = self.reward_model(**tokens).logits

        return logits.squeeze().item()

    # -------------------------
    # Rule-based helpers
    # -------------------------
    @staticmethod
    def extract_boxed(text: str) -> list[str]:
        matches = BOX_PATTERN.findall(text or "")
        return [m.strip() for m in matches]

    @staticmethod
    def _normalize(ans: Optional[str]) -> Optional[str]:
        if ans is None:
            return None
        ans = ans.strip().lower()
        ans = re.sub(r"\\,|\\!", "", ans)
        return re.sub(r"\s+", "", ans)

    @classmethod
    def _answers_match(cls, pred: Optional[str], truth: Optional[str]) -> bool:
        if pred is None or truth is None:
            return False
        pred_norm = cls._normalize(pred)
        truth_norm = cls._normalize(truth)
        if pred_norm == truth_norm:
            return True

        try:
            return abs(float(pred_norm) - float(truth_norm)) <= 1e-6
        except Exception:
            if SYMPY_AVAILABLE:
                try:
                    return sp.simplify(sp.sympify(pred) - sp.sympify(truth)) == 0
                except Exception:
                    return False
            return False

    # -------------------------
    # Composite reward
    # -------------------------
    def score(
        self,
        prompt: str,
        response: str,
        truth: Optional[str],
        rm_score: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute reward components and total score.

        Returns:
            dict with keys: rm, format, correct, brevity, total
        """
        rm_value = rm_score if rm_score is not None else self.score_reward_model(
            prompt, response
        )

        boxed_pred = self.extract_boxed(response)
        boxed_truth = self.extract_boxed(truth or "")
        has_box = bool(boxed_pred)
        format_reward = self.weights.format if has_box else -self.weights.format

        correct_reward = 0.0
        if has_box and boxed_truth:
            for candidate in reversed(boxed_pred):
                if self._answers_match(candidate, boxed_truth[-1]):
                    correct_reward = self.weights.correct
                    break

        brevity_reward = 0.0
        if correct_reward > 0:
            try:
                length = len(
                    self.tokenizer.encode(response, add_special_tokens=False)
                )
            except Exception:
                length = len(response.split())

            over = max(0, length - self.max_tokens_hint)
            brevity_reward = self.weights.brevity - 0.001 * over
            brevity_reward = max(min(brevity_reward, self.weights.brevity), -self.weights.brevity)

        total = (
            correct_reward
            + format_reward
            + brevity_reward
            + self.weights.reward_model * rm_value
        )

        return {
            "rm": rm_value,
            "format": format_reward,
            "correct": correct_reward,
            "brevity": brevity_reward,
            "total": total,
        }

