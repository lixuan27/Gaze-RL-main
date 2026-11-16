import pytest
import torch

from gaze_rl import group_advantages, landed_advantages


def test_binary_rewards():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = group_advantages(rewards, group_size=4, clip=None)
    assert torch.allclose(adv[:4], torch.tensor([0.5, -0.5, 0.5, -0.5]) / (0.5 + 1e-3))
    assert torch.equal(adv[4:], torch.zeros(4))


def test_delta_is_added_to_the_deviation():
    # One rollout differs by 0.01, e.g. a slightly better format score.
    rewards = torch.tensor([0.91, 0.90, 0.90, 0.90])
    std = rewards.std(unbiased=False)
    adv = group_advantages(rewards, group_size=4, clip=None)
    assert torch.allclose(adv, (rewards - rewards.mean()) / (std + 1e-3))
    assert adv[0].item() > 1.0

    # All eight answers correct, one with a better format score: clipped at 2.5.
    clipped = group_advantages(torch.tensor([1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]), group_size=8)
    assert clipped[0].item() == pytest.approx(2.5)


def test_groups_are_consecutive():
    rewards = torch.rand(12)
    adv = group_advantages(rewards, group_size=4, clip=None)
    for g in range(3):
        assert abs(adv[4 * g : 4 * g + 4].mean().item()) < 1e-6
    with pytest.raises(ValueError):
        group_advantages(torch.rand(10), group_size=4)


def test_landed_advantages_keep_rollout_credit():
    weights = torch.tensor([[0.0, 1.6, 0.4, 1.0], [0.0, 0.0, 1.3, 0.7]], requires_grad=True)
    adv = torch.tensor([0.8, -1.2])
    landed = landed_advantages(adv, weights)
    lengths = (weights > 0).sum(-1)
    assert torch.allclose(landed.sum(-1), adv * lengths)
    assert not landed.requires_grad
