from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LandingConfig:
    """Hyperparameters of the reward landing field.

    The defaults are the values used in the paper for every backbone, modality and
    benchmark. ``layers`` are 1-based decoder layers, so they index a Hugging Face
    ``hidden_states`` tuple directly (entry 0 is the embedding output). The default
    layer set is the one selected for Qwen2.5-VL-7B; set it explicitly for other
    decoders.

    Attributes:
        layers: decoder layers whose pre-action states are averaged.
        gamma: landing strength. ``0`` gives the uniform field, i.e. plain GRPO.
        eta: modulation strength, the bound of the tanh exponent over the centered
            log of the supported-branching score; any two weights of the field
            differ by at most a factor ``exp(2 * eta)``.
        kappa: temperature of the causal support kernel.
        lag: log-distance penalty of the support kernel.
        chi: share of a position's support that comes from the earlier states it
            reads through the support kernel.
        eps: stabilizer of the state and rollout normalizations.
    """

    layers: tuple[int, ...] = (14, 18, 22, 26)
    gamma: float = 0.70
    eta: float = 0.25
    kappa: float = 0.5
    lag: float = 0.1
    chi: float = 0.25
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not self.layers or min(self.layers) < 1 or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be distinct decoder layer indices >= 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.eta < 0 or self.lag < 0:
            raise ValueError("eta and lag must be non-negative")
        if self.kappa <= 0 or self.eps <= 0:
            raise ValueError("kappa and eps must be positive")
        if not 0.0 <= self.chi < 1.0:
            raise ValueError("chi must lie in [0, 1)")
