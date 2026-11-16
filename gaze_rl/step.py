"""Glue between the landing field and a GRPO-style update.

``prepare_rollouts`` runs once per batch of sampled rollouts, with the policy that
generated them; ``gaze_rl_loss`` is then called for every optimization step on that
batch. Both only need models that return ``logits`` and, for the policy,
``hidden_states``, as Hugging Face models do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from .advantages import group_advantages, landed_advantages
from .config import LandingConfig
from .landing import LandingSignals, landing_field, token_entropy
from .loss import policy_loss, token_logprobs

__all__ = ["Rollouts", "prepare_rollouts", "gaze_rl_loss"]


@dataclass
class Rollouts:
    """Detached quantities of one batch of grouped rollouts."""

    logp_old: torch.Tensor
    logp_ref: torch.Tensor
    weights: torch.Tensor
    advantages: torch.Tensor
    token_advantages: torch.Tensor
    response_mask: torch.Tensor
    signals: Optional[LandingSignals] = None


@torch.no_grad()
def prepare_rollouts(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    vision_mask: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    config: LandingConfig = LandingConfig(),
    advantage_clip: float | None = 2.5,
    keep_signals: bool = False,
) -> Rollouts:
    """Score sampled rollouts and build their per-token advantages.

    Args:
        policy: the policy that sampled the rollouts, in the mode used for sampling.
        reference: the frozen reference policy.
        inputs: model inputs of the full sequences (prompt, visual tokens and
            response), including ``input_ids``.
        vision_mask: ``[B, S]`` image and video tokens.
        response_mask: ``[B, S]`` sampled tokens.
        rewards: ``[B]`` verifier rewards; the rollouts of one prompt are
            consecutive, ``group_size`` at a time.
        config: landing hyperparameters; ``LandingConfig(gamma=0.0)`` gives GRPO.
        advantage_clip: bound on the group-normalized advantages, or ``None``.
        keep_signals: keep the per-token landing signals for logging.
    """
    out = policy(**inputs, output_hidden_states=True, use_cache=False)
    field = landing_field(
        out.hidden_states, token_entropy(out.logits), vision_mask, response_mask, config, return_signals=keep_signals
    )
    weights, signals = field if keep_signals else (field, None)
    logp_old = token_logprobs(out.logits, inputs["input_ids"])
    del out
    logp_ref = token_logprobs(reference(**inputs, use_cache=False).logits, inputs["input_ids"])
    advantages = group_advantages(rewards, group_size, clip=advantage_clip).to(weights.device)
    return Rollouts(
        logp_old=logp_old,
        logp_ref=logp_ref,
        weights=weights,
        advantages=advantages,
        token_advantages=landed_advantages(advantages, weights),
        response_mask=response_mask.bool(),
        signals=signals,
    )


def gaze_rl_loss(
    policy: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    rollouts: Rollouts,
    clip_eps: float = 0.2,
    beta: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss of one optimization step on prepared rollouts."""
    logp = token_logprobs(policy(**inputs, use_cache=False).logits, inputs["input_ids"])
    return policy_loss(
        logp,
        rollouts.logp_old,
        rollouts.logp_ref,
        rollouts.token_advantages,
        rollouts.response_mask,
        clip_eps=clip_eps,
        beta=beta,
    )
