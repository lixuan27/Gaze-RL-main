from __future__ import annotations

import torch

__all__ = ["group_advantages", "landed_advantages"]


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    delta: float = 1e-3,
    clip: float | None = 2.5,
) -> torch.Tensor:
    """Group-relative advantages ``(r - mean) / (std + delta)``.

    The rollouts of one prompt must be consecutive, ``group_size`` at a time, and
    ``std`` is the population standard deviation of the group. A group whose
    rollouts all receive the same reward gets zero advantage. ``delta`` is added
    to the deviation, not used as a lower bound, so a group whose rewards differ
    only slightly, for instance through a graded format score, is still scaled
    to large values; ``clip`` bounds those groups together with outliers.
    """
    if group_size < 1 or rewards.numel() % group_size:
        raise ValueError("the number of rollouts must be a positive multiple of group_size")
    grouped = rewards.float().view(-1, group_size)
    centered = grouped - grouped.mean(1, keepdim=True)
    advantages = centered / (grouped.std(1, unbiased=False, keepdim=True) + delta)
    if clip is not None:
        advantages = advantages.clamp(-clip, clip)
    return advantages.view(-1)


def landed_advantages(advantages: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Spread each rollout's advantage over its tokens with the landing weights.

    Because the weights have mean one over the response, every rollout keeps its
    total credit, ``advantage * response_length``. The weights are detached.
    """
    return advantages.float()[:, None] * weights.detach().float()
