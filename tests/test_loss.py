import torch

from gaze_rl import landed_advantages, policy_loss, sampled_kl, token_logprobs


def test_token_logprobs_are_shifted():
    logits = torch.randn(2, 5, 7)
    ids = torch.randint(0, 7, (2, 5))
    logp = token_logprobs(logits, ids)
    assert torch.all(logp[:, 0] == 0)
    for row in range(2):
        for p in range(1, 5):
            assert torch.allclose(logp[row, p], logits[row, p - 1].log_softmax(-1)[ids[row, p]])


def test_sampled_kl():
    a = torch.randn(4, 6)
    assert torch.allclose(sampled_kl(a, a), torch.zeros(4, 6))
    assert torch.all(sampled_kl(a, torch.randn(4, 6)) >= 0)
    far = sampled_kl(torch.zeros(3), torch.tensor([-50.0, 0.0, 50.0]))
    assert torch.isfinite(far).all()


def test_uniform_weights_give_grpo_loss():
    torch.manual_seed(0)
    logp = torch.randn(3, 6, requires_grad=True)
    old = logp.detach() + 0.05 * torch.randn(3, 6)
    ref = logp.detach() + 0.1 * torch.randn(3, 6)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, 0] = False
    mask[2, 4:] = False
    adv = torch.tensor([1.0, -0.5, 0.3])
    weights = mask.float().requires_grad_()

    loss, stats = policy_loss(logp, old, ref, landed_advantages(adv, weights), mask)
    ratio = torch.exp(logp - old)
    surrogate = torch.minimum(ratio * adv[:, None], ratio.clamp(0.8, 1.2) * adv[:, None])
    expected = -(((surrogate - 1e-3 * sampled_kl(logp, ref)) * mask).sum(-1) / mask.sum(-1)).mean()
    assert torch.allclose(loss, expected)
    assert set(stats) == {"kl", "clip_fraction"}

    loss.backward()
    assert weights.grad is None
    assert torch.all(logp.grad[~mask] == 0)
