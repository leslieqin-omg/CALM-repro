from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<generate-antibody>", "<generate-epitope>")
TASKS = ("generate-antibody", "generate-epitope")


class SequenceTokenizer:
    def __init__(self) -> None:
        self.tokens = list(SPECIAL_TOKENS) + list(AMINO_ACIDS)
        self.index = {token: position for position, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.index["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.index["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.index["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.index["<unk>"]

    def task_id(self, task: str) -> int:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task}")
        return self.index[f"<{task}>"]

    def prefix_ids(self, task: str) -> list[int]:
        return [self.task_id(task), self.bos_id]

    def encode(self, sequence: str, task: str) -> list[int]:
        body = [self.index.get(residue, self.unk_id) for residue in sequence]
        return self.prefix_ids(task) + body + [self.eos_id]

    def decode(self, ids) -> str:
        residues = []
        for value in ids:
            token = self.tokens[int(value)]
            if token == "<eos>":
                break
            if len(token) == 1:
                residues.append(token)
        return "".join(residues)


@dataclass
class DecoderConfig:
    hidden_dim: int = 512
    layers: int = 4
    heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_len: int = 550


def ragged_to_padded(tokens: torch.Tensor, index: torch.Tensor, batch_size: int):
    lengths = torch.bincount(index, minlength=batch_size)
    width = int(lengths.max().item()) if lengths.numel() else 0
    padded = tokens.new_zeros(batch_size, max(width, 1), tokens.shape[1])
    pad_mask = torch.ones(batch_size, max(width, 1), dtype=torch.bool, device=tokens.device)
    if index.numel() == 0:
        pad_mask[:, 0] = False
        return padded, pad_mask
    offsets = torch.cumsum(lengths, 0) - lengths
    positions = torch.arange(index.shape[0], device=index.device) - offsets[index]
    padded[index, positions] = tokens
    pad_mask[index, positions] = False
    return padded, pad_mask


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, memory, memory_padding_mask, causal_mask):
        normed = self.self_norm(hidden)
        attended, _weights = self.self_attention(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)
        normed = self.cross_norm(hidden)
        crossed, _cross_weights = self.cross_attention(
            normed,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(crossed)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class CalmDecoder(nn.Module):
    def __init__(self, config: DecoderConfig, memory_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_len + 8, config.hidden_dim)
        self.memory_projection = nn.Linear(memory_dim, config.hidden_dim)
        self.memory_norm = nn.LayerNorm(config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(config.hidden_dim, config.heads, config.ffn_dim, config.dropout)
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, tokens, memory, memory_padding_mask=None):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        context = self.memory_norm(self.memory_projection(memory))
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, context, memory_padding_mask, causal)
        return self.output(self.final_norm(hidden))

    def generate(self, memory, task: str, tokenizer: SequenceTokenizer, max_len=None, memory_padding_mask=None):
        self.eval()
        batch = int(memory.shape[0])
        limit = int(max_len or self.config.max_len)
        prefix = tokenizer.prefix_ids(task)
        tokens = torch.tensor([prefix for _ in range(batch)], dtype=torch.long, device=memory.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        with torch.no_grad():
            for _step in range(limit):
                logits = self.forward(tokens, memory, memory_padding_mask)
                choice = logits[:, -1].argmax(dim=-1)
                choice = torch.where(finished, torch.full_like(choice, tokenizer.pad_id), choice)
                tokens = torch.cat([tokens, choice.unsqueeze(1)], dim=1)
                finished = finished | (choice == tokenizer.eos_id)
                if bool(finished.all()):
                    break
        return tokens


def teacher_forcing(tokenizer: SequenceTokenizer, sequences, task: str, device):
    encoded = [tokenizer.encode(sequence, task) for sequence in sequences]
    width = max(len(item) for item in encoded)
    inputs = torch.full((len(encoded), width - 1), tokenizer.pad_id, dtype=torch.long)
    targets = torch.full((len(encoded), width - 1), tokenizer.pad_id, dtype=torch.long)
    for row, item in enumerate(encoded):
        span = len(item) - 1
        inputs[row, :span] = torch.tensor(item[:-1], dtype=torch.long)
        targets[row, :span] = torch.tensor(item[1:], dtype=torch.long)
    return inputs.to(device), targets.to(device)
