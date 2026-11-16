import math

import pytest
import torch
import torch.nn.functional as F

from gaze_rl import LandingConfig, landing_field, landing_weights, response_positions, support_coefficient
from toy import random_rollouts

CFG = LandingConfig(layers=(2, 4, 5))


def reference_weights(hidden, entropy, vision, response, cfg):
    """Slow per-token implementation for a single rollout."""
    eps = cfg.eps
    positions = [p for p in range(len(response)) if response[p]]
    T = len(positions)
    anchoring, states = [], []
    for p in positions:
        a, s = 0.0, 0.0
        for layer in cfg.layers:
            h = hidden[layer]
            prototype = F.normalize(h[vision], dim=-1, eps=eps).mean(0)
            a += float(F.normalize(h[p - 1], dim=-1, eps=eps) @ prototype)
            s = s + h[p - 1]
        anchoring.append(a / len(cfg.layers))
        s = s / len(cfg.layers)
        states.append(s / (s.norm() + eps))
    anchoring = torch.tensor(anchoring)

    def scale(x):
        return (x - x.min()) / (x.max() - x.min() + eps)

    a_hat = scale(anchoring)
    support = torch.zeros(T)
    for t in range(T):
        support[t] = (1 - cfg.chi) * a_hat[t]
        if t > 0:
            scores = torch.tensor(
                [float(states[k] @ states[t]) / cfg.kappa - cfg.lag * math.log(1 + t - k) for k in range(t)]
            )
            support[t] += cfg.chi * float(scores.softmax(0) @ support[:t])
    branching = torch.tensor([float(entropy[p - 1]) for p in positions]) * support
    log_evidence = branching.clamp_min(1e-12).log()
    q = torch.exp(cfg.eta * torch.tanh(log_evidence - log_evidence.mean()))
    return (1 - cfg.gamma) + cfg.gamma * T * q / q.sum()


def packed_signals(anchoring, seed=0, dim=6):
    g = torch.Generator().manual_seed(seed)
    T = anchoring.shape[-1]
    states = F.normalize(torch.randn(1, T, dim, generator=g), dim=-1)
    entropy = torch.rand(1, T, generator=g) * 4
    return anchoring.view(1, T), states, entropy, torch.ones(1, T, dtype=torch.bool)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_implementation(seed):
    hidden, entropy, vision, response = random_rollouts(seed)
    weights = landing_field(hidden, entropy, vision, response, CFG)
    for row in range(response.shape[0]):
        expected = reference_weights([h[row] for h in hidden], entropy[row], vision[row], response[row], CFG)
        assert torch.allclose(weights[row][response[row]], expected, atol=1e-5)
        assert torch.all(weights[row][~response[row]] == 0)


@pytest.mark.parametrize("gamma", [0.3, 0.7, 1.0])
def test_weights_have_mean_one_and_floor(gamma):
    cfg = LandingConfig(layers=CFG.layers, gamma=gamma)
    hidden, entropy, vision, response = random_rollouts(seed=3)
    weights = landing_field(hidden, entropy, vision, response, cfg)
    for row in range(response.shape[0]):
        w = weights[row][response[row]]
        assert abs(w.mean().item() - 1) < 1e-5
        assert w.min().item() >= 1 - gamma - 1e-6


def test_zero_gamma_is_uniform():
    hidden, entropy, vision, response = random_rollouts(seed=4)
    weights = landing_field(hidden, entropy, vision, response, LandingConfig(layers=CFG.layers, gamma=0.0))
    assert torch.equal(weights, response.float())


def test_independent_of_batch_and_padding():
    hidden, entropy, vision, response = random_rollouts(seed=5)
    full = landing_field(hidden, entropy, vision, response, CFG)
    alone = landing_field([h[1:2] for h in hidden], entropy[1:2], vision[1:2], response[1:2], CFG)
    assert torch.allclose(full[1:2], alone, atol=1e-6)

    pad = 5
    shifted = landing_field(
        [F.pad(h, (0, 0, pad, 0)) for h in hidden],
        F.pad(entropy, (pad, 0)),
        F.pad(vision, (pad, 0)),
        F.pad(response, (pad, 0)),
        CFG,
    )
    assert torch.allclose(shifted[:, pad:], full, atol=1e-6)


def test_support_only_reads_earlier_positions():
    g = torch.Generator().manual_seed(6)
    states = F.normalize(torch.randn(1, 10, 5, generator=g), dim=-1)
    anchoring = torch.rand(1, 10, generator=g)
    valid = torch.ones(1, 10, dtype=torch.bool)
    before = support_coefficient(states, anchoring, valid)
    states[:, 6:] = F.normalize(torch.randn(1, 4, 5, generator=g), dim=-1)
    anchoring[:, 6:] = torch.rand(1, 4, generator=g)
    after = support_coefficient(states, anchoring, valid)
    assert torch.allclose(before[:, :6], after[:, :6])
    assert not torch.allclose(before[:, 6:], after[:, 6:])
    assert torch.all((before >= 0) & (before <= 1))
    assert torch.allclose(support_coefficient(states, anchoring, valid, chi=0.0), anchoring)

    # a position that is not anchored itself is supported through the anchored states it reads
    first_only = torch.zeros_like(anchoring)
    first_only[0, 0] = 1.0
    assert torch.all(support_coefficient(states, first_only, valid)[0, 1:] > 0)


@pytest.mark.parametrize("level", [-0.4, 0.0, 0.2, 0.9])
def test_constant_anchoring_gives_uniform_field(level):
    out = landing_weights(*packed_signals(torch.full((12,), level)), LandingConfig(gamma=1.0))
    assert torch.all(out["support"] == 0)
    assert torch.allclose(out["weights"], torch.ones(1, 12))


@pytest.mark.parametrize("noise", [1e-7, 1e-3, 0.3])
def test_field_orders_positions_by_branching_and_is_bounded(noise):
    cfg = LandingConfig(gamma=1.0)
    g = torch.Generator().manual_seed(7)
    anchoring = 0.5 + noise * torch.randn(16, generator=g)
    out = landing_weights(*packed_signals(anchoring, seed=8), cfg)
    branching, field = out["branching"][0].double(), out["field"][0].double()
    order = branching.argsort()
    assert torch.all(field[order][1:] >= field[order][:-1] - 1e-7)
    assert math.log(field.max() / field.min()) <= 2 * cfg.eta + 1e-6
    assert abs(field.mean().item() - 1) < 1e-5


@pytest.mark.parametrize("seed", range(3))
def test_nearly_constant_anchoring_from_hidden_states(seed):
    # Every pre-action state has cosine 0.5 to the single visual state, up to float32 rounding.
    T, layers = 6, (1, 2, 3)
    g = torch.Generator().manual_seed(seed)
    others = F.normalize(torch.randn(T, 3, generator=g), dim=-1)
    h = torch.zeros(T + 2, 4)
    h[0, 0] = 1.0
    h[1 : T + 1] = torch.cat([torch.full((T, 1), 0.5), math.sqrt(0.75) * others], -1)
    hidden = tuple(h[None].clone() for _ in range(len(layers) + 1))
    vision = torch.zeros(1, T + 2, dtype=torch.bool)
    vision[0, 0] = True
    response = torch.zeros(1, T + 2, dtype=torch.bool)
    response[0, 2:] = True
    entropy = torch.zeros(1, T + 2)
    entropy[0, 1 : T + 1] = torch.tensor([0.1, 3.0, 0.5, 2.0, 0.2, 4.0])

    cfg = LandingConfig(layers=layers, gamma=1.0)
    _, signals = landing_field(hidden, entropy, vision, response, cfg, return_signals=True)
    anchoring = signals.anchoring[0]
    assert float(anchoring.max() - anchoring.min()) < 1e-6
    # the field stays mean one and its spread is bounded by the tanh exponent
    field = signals.field[0, : T]
    assert abs(field.mean().item() - 1) < 1e-5
    assert math.log(field.max() / field.min()) <= 2 * cfg.eta + 1e-6


def test_short_and_empty_responses():
    hidden, entropy, vision, response = random_rollouts(seed=10)
    response[0] = False
    response[1] = False
    response[1, 20] = True
    weights = landing_field(hidden, entropy, vision, response, CFG)
    assert torch.isfinite(weights).all()
    assert torch.all(weights[0] == 0)
    assert weights[1, 20].item() == pytest.approx(1.0)
    assert weights[1].sum().item() == pytest.approx(1.0)


def test_bfloat16_hidden_states():
    hidden, entropy, vision, response = random_rollouts(seed=11)
    low = tuple(h.bfloat16() for h in hidden)
    weights = landing_field(low, entropy.bfloat16(), vision, response, CFG)
    expected = landing_field([h.float() for h in low], entropy.bfloat16().float(), vision, response, CFG)
    assert weights.dtype == torch.float32
    assert torch.allclose(weights, expected, atol=1e-5)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        response_positions(torch.tensor([[True, True, False]]))
    hidden, entropy, vision, response = random_rollouts(seed=12)
    with pytest.raises(ValueError):
        landing_field(hidden, entropy, vision, response, LandingConfig(layers=(2, 7)))
    for bad in (dict(gamma=1.5), dict(chi=1.0), dict(chi=-0.1), dict(kappa=0.0)):
        with pytest.raises(ValueError):
            LandingConfig(**bad)
