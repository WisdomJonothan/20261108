"""Low-rank updates for the last layers of the Qwen3 relevance scorer."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, scale):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.down, a=5 ** .5)
        self.scale = scale / rank

    def forward(self, x):
        update = F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype))
        return self.base(x) + self.scale * update


def install(model, layers, rank, scale):
    if not 0 < layers <= len(model.model.layers) or rank <= 0 or scale <= 0:
        raise ValueError('Invalid low-rank configuration')
    model.requires_grad_(False)
    for block in model.model.layers[-layers:]:
        for parent, name in ((block.self_attn, 'q_proj'), (block.self_attn, 'v_proj'),
                             (block.mlp, 'down_proj')):
            setattr(parent, name, LowRankLinear(getattr(parent, name), rank, scale))


def save_trainable(model, path, config, training):
    path = Path(path)
    with path.open('xb') as stream:
        torch.save({'config': config, 'training': training,
                    'weights': {name: value.detach().cpu() for name, value in model.named_parameters()
                                if value.requires_grad}}, stream)


def load_adapter(model, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    install(model, **saved['config'])
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if current.keys() != saved['weights'].keys():
        raise ValueError('Adapter parameters and configured layers differ')
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved['weights'][name])
    return saved['training']
