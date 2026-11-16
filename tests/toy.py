"""Small synthetic inputs shared by the tests."""

from types import SimpleNamespace

import torch
from torch import nn


def random_rollouts(seed=0, batch=3, length=48, dim=12, layers=6, starts=(10, 14, 20), ends=(40, 30, 47)):
    """Random hidden states and entropies with visual tokens at positions 1-7."""
    g = torch.Generator().manual_seed(seed)
    hidden = tuple(torch.randn(batch, length, dim, generator=g) + 0.4 for _ in range(layers + 1))
    entropy = torch.rand(batch, length, generator=g) * 4
    vision = torch.zeros(batch, length, dtype=torch.bool)
    vision[:, 1:8] = True
    response = torch.zeros(batch, length, dtype=torch.bool)
    for row in range(batch):
        response[row, starts[row] : ends[row]] = True
    return hidden, entropy, vision, response


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=32, dim=16, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, 2, 2 * dim, dropout=0.0, batch_first=True) for _ in range(layers)
        )
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, output_hidden_states=False, **_):
        h = self.embed(input_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1])
        states = [h]
        for block in self.blocks:
            h = block(h, src_mask=mask, is_causal=True)
            states.append(h)
        return SimpleNamespace(logits=self.head(h), hidden_states=tuple(states) if output_hidden_states else None)
