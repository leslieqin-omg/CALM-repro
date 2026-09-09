from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


def pooled_projection(
    head: ProjectionHead,
    tokens: torch.Tensor,
    index: torch.Tensor,
    batch_size: int,
    token_chunk: int,
    use_checkpoint: bool,
) -> torch.Tensor:
    pooled = torch.zeros(batch_size, head.out_dim, dtype=torch.float32, device=tokens.device)
    step = max(1, int(token_chunk))
    for start in range(0, tokens.shape[0], step):
        block = tokens[start:start + step]
        if use_checkpoint:
            projected = checkpoint(head, block, use_reentrant=False)
        else:
            projected = head(block)
        pooled = pooled.index_add(0, index[start:start + step], projected.float())
    counts = torch.bincount(index, minlength=batch_size).clamp(min=1).unsqueeze(1).to(torch.float32)
    return pooled / counts


class CALM(nn.Module):
    def __init__(self, config: ModelConfig, antibody_dim: int, antigen_dim: int) -> None:
        super().__init__()
        self.config = config
        self.antibody_head = ProjectionHead(antibody_dim, config.hidden_dim, config.embed_dim, config.dropout)
        self.antigen_head = ProjectionHead(antigen_dim, config.hidden_dim, config.embed_dim, config.dropout)
        initial = math.log(max(float(config.temperature), 1e-4))
        tensor = torch.tensor(initial, dtype=torch.float32)
        self.log_temperature = nn.Parameter(tensor, requires_grad=bool(config.learnable_temperature))
        self.token_chunk = 1 << 24
        self.use_checkpoint = False

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 1.0)

    def configure_runtime(self, token_chunk: int, use_checkpoint: bool) -> None:
        self.token_chunk = int(token_chunk)
        self.use_checkpoint = bool(use_checkpoint)

    def encode_antibody(self, tokens: torch.Tensor, index: torch.Tensor, batch_size: int) -> torch.Tensor:
        active = self.use_checkpoint and self.training
        pooled = pooled_projection(self.antibody_head, tokens, index, batch_size, self.token_chunk, active)
        return F.normalize(pooled, dim=-1)

    def encode_antigen(self, tokens: torch.Tensor, index: torch.Tensor, batch_size: int) -> torch.Tensor:
        active = self.use_checkpoint and self.training
        pooled = pooled_projection(self.antigen_head, tokens, index, batch_size, self.token_chunk, active)
        return F.normalize(pooled, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = int(batch["size"].item())
        antibody = self.encode_antibody(batch["antibody_tokens"], batch["antibody_index"], size)
        antigen = self.encode_antigen(batch["antigen_tokens"], batch["antigen_index"], size)
        logits = antibody @ antigen.t() / self.temperature
        return antibody, antigen, logits
