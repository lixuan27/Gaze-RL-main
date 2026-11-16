from __future__ import annotations

import torch

__all__ = ["token_logprobs", "sampled_kl", "policy_loss"]


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Log-probability of every token under the distribution at the previous position.

    Position 0 has no previous position and is set to zero. Rows are processed one
    at a time to keep the float32 log-softmax over the vocabulary small.
    """
    logp = torch.zeros(input_ids.shape, dtype=torch.float32, device=logits.device)
    target = input_ids[:, 1:]
    for row in range(logits.shape[0]):
        row_logp = logits[row, :-1].float().log_softmax(-1)
        logp[row, 1:] = row_logp.gather(-1, target[row, :, None]).squeeze(-1)
    return logp


def sampled_kl(logp: torch.Tensor, logp_ref: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    """k3 estimate of ``KL(pi || pi_ref)`` from the log-probabilities of sampled tokens.

    With ``x = logp_ref - logp`` the estimate is ``exp(x) - x - 1``. It is
    non-negative, and unbiased when the tokens are sampled from ``pi``. ``x`` is
    clamped to ``[-clip, clip]`` in the forward pass only, so a rare extreme token
    cannot overflow the loss.
    """
    x = logp_ref.float() - logp.float()
    x = x + (x.clamp(-clip, clip) - x).detach()
    return torch.expm1(x) - x


def policy_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    token_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate on per-token advantages with a KL penalty to the reference.

    Each term is averaged over the response tokens of its rollout and then over the
    rollouts. With uniform weights, ``token_advantages`` repeats the rollout
    advantage and this is the GRPO loss.

    Returns:
        The loss to minimize and a few detached statistics.
    """
    mask = response_mask.float()
    length = mask.sum(-1).clamp_min(1.0)
    ratio = torch.exp(logp - logp_old.detach())
    surrogate = torch.minimum(ratio * token_advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * token_advantages)
    kl = sampled_kl(logp, logp_ref.detach())
    objective = ((surrogate - beta * kl) * mask).sum(-1) / length
    with torch.no_grad():
        clipped = ((ratio - 1).abs() > clip_eps).float()
        stats = {
            "kl": ((kl * mask).sum(-1) / length).mean().item(),
            "clip_fraction": ((clipped * mask).sum() / mask.sum().clamp_min(1.0)).item(),
        }
    return -objective.mean(), stats
