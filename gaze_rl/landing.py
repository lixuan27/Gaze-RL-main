"""Reward landing field, computed from one forward pass of the rollout-time policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .config import LandingConfig

__all__ = [
    "LandingSignals",
    "token_entropy",
    "response_positions",
    "visual_anchoring",
    "support_coefficient",
    "landing_weights",
    "landing_field",
]


@dataclass
class LandingSignals:
    """Per-token quantities behind a landing field.

    Responses are packed to the left: every tensor is ``[B, T]`` with ``T`` the
    longest response in the batch, ``valid`` marks real response tokens and
    ``index`` holds their positions in the full sequence.
    """

    index: torch.Tensor
    valid: torch.Tensor
    anchoring: torch.Tensor
    support: torch.Tensor
    branching: torch.Tensor
    field: torch.Tensor
    weights: torch.Tensor


@torch.no_grad()
def token_entropy(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Entropy of the next-token distribution at every position, in float32.

    The full-vocabulary softmax is taken ``chunk_size`` positions at a time, which
    keeps the float32 copy of the logits small for long sequences.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
    for start in range(0, flat.shape[0], chunk_size):
        logp = flat[start : start + chunk_size].float().log_softmax(-1)
        out[start : start + chunk_size] = -(logp.exp() * logp).sum(-1)
    return out.view(logits.shape[:-1])


def response_positions(response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequence positions of the response tokens of each row, packed to the left.

    Returns ``(index, valid)``. Padding slots of ``index`` point at position 1, so
    ``index - 1`` is always a valid pre-action position.
    """
    if bool(response_mask[:, 0].any()):
        raise ValueError("a response token cannot sit at position 0: it has no pre-action state")
    batch, length = response_mask.shape
    counts = response_mask.sum(1)
    positions = torch.arange(length, device=response_mask.device).expand(batch, length)
    width = int(counts.max().clamp_min(1))
    index = torch.where(response_mask, positions, positions + length).argsort(dim=1)[:, :width]
    valid = torch.arange(width, device=index.device)[None] < counts[:, None]
    return index.masked_fill(~valid, 1), valid


def _gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.gather(1, index)
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


def _rollout_norm(x: torch.Tensor, valid: torch.Tensor, eps: float) -> torch.Tensor:
    # min-max scaling over the valid tokens of each row
    big = torch.finfo(x.dtype).max
    lo = x.masked_fill(~valid, big).amin(-1, keepdim=True)
    hi = x.masked_fill(~valid, -big).amax(-1, keepdim=True)
    return ((x - lo) / ((hi - lo).clamp_min(0.0) + eps)).masked_fill(~valid, 0.0)


@torch.no_grad()
def visual_anchoring(
    hidden_states: Sequence[torch.Tensor],
    vision_mask: torch.Tensor,
    pre_index: torch.Tensor,
    layers: Sequence[int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Visual anchoring of the states that predict the response tokens.

    At each layer, the visual prototype is the mean of the unit-normalized states
    of the visual tokens, and the anchoring of a response token is the cosine
    similarity between the prototype and the state at ``pre_index``, the state from
    which the token was sampled. Both are averaged over ``layers``.

    Returns:
        ``(anchoring, states)``: ``[B, T]`` anchoring, and the layer-averaged
        pre-action states rescaled to unit norm, ``[B, T, D]``.
    """
    if max(layers) >= len(hidden_states):
        raise ValueError(f"layer {max(layers)} is out of range for {len(hidden_states) - 1} decoder layers")
    weight = vision_mask.float()
    count = weight.sum(1, keepdim=True).clamp_min(1.0)
    anchoring = 0.0
    states = 0.0
    for layer in layers:
        h = hidden_states[layer].float()
        prototype = torch.einsum("bs,bsd->bd", weight, F.normalize(h, dim=-1, eps=eps)) / count
        pre = _gather(h, pre_index)
        anchoring = anchoring + (F.normalize(pre, dim=-1, eps=eps) * prototype[:, None]).sum(-1)
        states = states + pre
    anchoring = anchoring / len(layers)
    states = states / len(layers)
    return anchoring, states / (states.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def support_coefficient(
    states: torch.Tensor,
    anchoring: torch.Tensor,
    valid: torch.Tensor,
    kappa: float = 0.5,
    lag: float = 0.1,
    chi: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Whether a position is visually anchored itself or draws on earlier states that are.

    A causal kernel ``K``, a softmax over earlier positions of state similarity
    divided by ``kappa`` minus ``lag * log(1 + distance)``, selects the predecessors
    of each position. The support solves ``z = (1 - chi) * anchoring + chi * K z``
    by forward substitution, so support propagates along chains of related states;
    it lies in ``[0, 1]`` and ``chi = 0`` reduces it to the anchoring itself.

    Args:
        states: ``[B, T, D]`` unit-norm pre-action states.
        anchoring: ``[B, T]`` anchoring, min-max normalized within each rollout.
        valid: ``[B, T]`` mask of real response tokens.
    """
    steps = torch.arange(states.shape[1], device=states.device)
    distance = steps[:, None] - steps[None, :]
    earlier = (distance > 0)[None] & valid[:, None, :] & valid[:, :, None]
    scores = states @ states.transpose(1, 2) / kappa - lag * torch.log1p(distance.clamp_min(0).float())
    kernel = scores.masked_fill(~earlier, float("-inf")).softmax(-1).nan_to_num(0.0).double()
    eye = torch.eye(kernel.shape[-1], dtype=torch.float64, device=kernel.device).expand_as(kernel)
    rhs = ((1.0 - chi) * anchoring.double() * valid)[..., None]
    return torch.linalg.solve_triangular(eye - chi * kernel, rhs, upper=False).squeeze(-1).float() * valid


@torch.no_grad()
def landing_weights(
    anchoring: torch.Tensor,
    states: torch.Tensor,
    entropy: torch.Tensor,
    valid: torch.Tensor,
    config: LandingConfig = LandingConfig(),
) -> dict[str, torch.Tensor]:
    """Landing weights from packed per-token signals.

    Args:
        anchoring: ``[B, T]`` raw visual anchoring of each response token.
        states: ``[B, T, D]`` unit-norm pre-action states.
        entropy: ``[B, T]`` entropy of the distribution each token was sampled from.
        valid: ``[B, T]`` mask of real response tokens.

    Returns:
        A dict with the blended ``weights`` (mean one over the valid tokens of every
        row with a response, zero elsewhere) and the intermediate ``support``,
        ``branching`` and unblended ``field``.
    """
    eps = config.eps
    anchoring = anchoring.float().masked_fill(~valid, 0.0)
    anchoring_hat = _rollout_norm(anchoring, valid, eps)
    support = support_coefficient(states.float(), anchoring_hat, valid, config.kappa, config.lag, config.chi, eps)
    branching = (entropy.float().clamp_min(0.0) * support).masked_fill(~valid, 0.0)

    # tanh-bounded exponent over the centered log-evidence; uniform when every score vanishes
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    log_evidence = branching.clamp_min(1e-12).log().masked_fill(~valid, 0.0)
    centered = log_evidence - log_evidence.sum(-1, keepdim=True) / count
    exponent = config.eta * torch.tanh(centered)
    field = (count * exponent.masked_fill(~valid, float("-inf")).softmax(-1)).nan_to_num(0.0)
    weights = ((1.0 - config.gamma) + config.gamma * field) * valid
    return {"weights": weights, "support": support * valid, "branching": branching, "field": field}


@torch.no_grad()
def landing_field(
    hidden_states: Sequence[torch.Tensor],
    entropy: torch.Tensor,
    vision_mask: torch.Tensor,
    response_mask: torch.Tensor,
    config: LandingConfig = LandingConfig(),
    return_signals: bool = False,
):
    """Per-token landing weights of a batch of rollouts.

    Args:
        hidden_states: per-layer states ``[B, S, D]`` of the rollout-time policy,
            embedding output first, as returned with ``output_hidden_states=True``.
        entropy: ``[B, S]`` entropy of the next-token distribution at each position,
            e.g. ``token_entropy(logits)``.
        vision_mask: ``[B, S]`` image and video tokens.
        response_mask: ``[B, S]`` sampled tokens. The token at position ``p`` is
            predicted from the state at ``p - 1``.
        config: landing hyperparameters.
        return_signals: also return a :class:`LandingSignals` for inspection.

    Returns:
        ``[B, S]`` float32 weights, zero outside the response and with mean one over
        the response tokens of each rollout; with ``return_signals``, a tuple of the
        weights and the signals.
    """
    index, valid = response_positions(response_mask.bool())
    pre_index = index - 1
    anchoring, states = visual_anchoring(hidden_states, vision_mask.bool(), pre_index, config.layers, config.eps)
    out = landing_weights(anchoring, states, _gather(entropy.float(), pre_index), valid, config)

    weights = torch.zeros(response_mask.shape, dtype=torch.float32, device=index.device)
    weights.scatter_add_(1, index, out["weights"])
    if not return_signals:
        return weights
    signals = LandingSignals(
        index=index,
        valid=valid,
        anchoring=anchoring.masked_fill(~valid, 0.0),
        support=out["support"],
        branching=out["branching"],
        field=out["field"],
        weights=out["weights"],
    )
    return weights, signals
