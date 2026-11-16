"""A few Gaze-RL updates of a tiny random causal model with a fake visual memory.

This shows how the pieces fit into a GRPO loop. The model, the "visual" tokens and
the reward are random, so the numbers say nothing about model quality.

    python examples/toy_update.py
"""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from gaze_rl import LandingConfig, gaze_rl_loss, prepare_rollouts


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=64, dim=32, layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.0, batch_first=True) for _ in range(layers)
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


@torch.no_grad()
def sample(model, prefix, steps):
    ids = prefix
    for _ in range(steps):
        logits = model(ids).logits[:, -1]
        ids = torch.cat([ids, torch.multinomial(logits.softmax(-1), 1)], 1)
    return ids


def main():
    torch.manual_seed(0)
    prompts, group_size, n_visual, n_text, n_response = 2, 4, 8, 6, 12
    policy = TinyCausalLM()
    reference = copy.deepcopy(policy).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    config = LandingConfig(layers=(2, 4, 6))

    prefix = torch.randint(0, 64, (prompts, n_visual + n_text)).repeat_interleave(group_size, 0)
    input_ids = sample(policy, prefix, n_response)
    vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    vision_mask[:, :n_visual] = True
    response_mask = torch.zeros_like(vision_mask)
    response_mask[:, n_visual + n_text :] = True
    rewards = (input_ids[:, -1] % 2 == 0).float()
    inputs = {"input_ids": input_ids}

    rollouts = prepare_rollouts(
        policy, reference, inputs, vision_mask, response_mask, rewards, group_size, config, keep_signals=True
    )
    weights = rollouts.signals.weights
    print("advantages      ", [round(a, 3) for a in rollouts.advantages.tolist()])
    print("weight per token  mean {:.3f}  min {:.3f}  max {:.3f}".format(weights.mean(), weights.min(), weights.max()))
    print("first rollout    ", " ".join(f"{w:.2f}" for w in weights[0].tolist()))

    for step in range(3):
        loss, stats = gaze_rl_loss(policy, inputs, rollouts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step}  loss {loss.item():+.4f}  kl {stats['kl']:.2e}  clipped {stats['clip_fraction']:.2f}")


if __name__ == "__main__":
    main()
