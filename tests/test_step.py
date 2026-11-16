import copy

import torch

from gaze_rl import LandingConfig, gaze_rl_loss, prepare_rollouts
from toy import TinyCausalLM


def toy_batch(groups=2, group_size=4, n_visual=6, n_text=4, n_response=8, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, 32, (groups, n_visual + n_text + n_response)).repeat_interleave(group_size, 0)
    ids[:, n_visual + n_text :] = torch.randint(0, 32, (groups * group_size, n_response))
    vision = torch.zeros_like(ids, dtype=torch.bool)
    vision[:, :n_visual] = True
    response = torch.zeros_like(vision)
    response[:, n_visual + n_text :] = True
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    return {"input_ids": ids}, vision, response, rewards


def test_update_runs_and_preserves_rollout_credit():
    policy = TinyCausalLM()
    reference = copy.deepcopy(policy).eval()
    inputs, vision, response, rewards = toy_batch()
    rollouts = prepare_rollouts(
        policy, reference, inputs, vision, response, rewards, 4, LandingConfig(layers=(1, 3, 4)), keep_signals=True
    )
    weights = rollouts.weights[response].view(8, -1)
    assert torch.allclose(weights.mean(-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(rollouts.token_advantages.sum(-1), rollouts.advantages * response.sum(-1), atol=1e-5)
    assert rollouts.signals is not None and rollouts.signals.weights.shape == (8, 8)

    # Before the first step the ratio is one and the policy equals the reference,
    # so the loss is minus the mean rollout advantage, whatever the weights are.
    loss, stats = gaze_rl_loss(policy, inputs, rollouts)
    assert torch.allclose(loss, -rollouts.advantages.mean(), atol=1e-6)
    assert stats["kl"] < 1e-8

    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-2)
    loss.backward()
    optimizer.step()
    loss, _ = gaze_rl_loss(policy, inputs, rollouts)
    assert torch.isfinite(loss)


def test_zero_gamma_matches_grpo():
    policy = TinyCausalLM()
    inputs, vision, response, rewards = toy_batch(seed=1)
    rollouts = prepare_rollouts(
        policy, policy, inputs, vision, response, rewards, 4, LandingConfig(layers=(1, 3, 4), gamma=0.0)
    )
    assert torch.equal(rollouts.weights, response.float())
    assert torch.allclose(rollouts.token_advantages, rollouts.advantages[:, None] * response.float())
