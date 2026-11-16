"""Gaze-RL: learning where rewards land in multimodal reasoning."""

from .advantages import group_advantages, landed_advantages
from .config import LandingConfig
from .landing import (
    LandingSignals,
    landing_field,
    landing_weights,
    response_positions,
    support_coefficient,
    token_entropy,
    visual_anchoring,
)
from .loss import policy_loss, sampled_kl, token_logprobs
from .step import Rollouts, gaze_rl_loss, prepare_rollouts

__version__ = "1.0.0"

__all__ = [
    "LandingConfig",
    "LandingSignals",
    "landing_field",
    "landing_weights",
    "response_positions",
    "support_coefficient",
    "token_entropy",
    "visual_anchoring",
    "group_advantages",
    "landed_advantages",
    "policy_loss",
    "sampled_kl",
    "token_logprobs",
    "Rollouts",
    "gaze_rl_loss",
    "prepare_rollouts",
]
