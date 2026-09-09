from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import schema
from .embeddings import RaggedEmbeddingStore

MASK_MODES = ("none", "interface")


@dataclass
class PairRecord:
    pair_id: str
    vh: str
    vl: str
    antigen: str
    vh_keep: np.ndarray
    vl_keep: np.ndarray
    antigen_keep: np.ndarray
    antibody_label: int
    antigen_label: int
    cluster_id: str


def _selection(sequence: str, mask: str, mask_mode: str) -> np.ndarray:
    if sequence == "":
        return np.zeros(0, dtype=bool)
    if mask_mode == "none":
        return np.ones(len(sequence), dtype=bool)
    return schema.mask_to_array(mask, len(sequence))


def _selected_residues(sequence: str, keep: np.ndarray) -> str:
    if sequence == "" or keep.size == 0:
        return ""
    codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return codes[keep].tobytes().decode("ascii")


def build_records(frame: pd.DataFrame, mask_mode: str) -> list[PairRecord]:
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode}")
    antibody_keys: dict[str, int] = dict()
    antigen_keys: dict[str, int] = dict()
    records: list[PairRecord] = []
    for row in frame.itertuples():
        vh_keep = _selection(row.Antibody_VH_AA, row.Antibody_VH_Paratope, mask_mode)
        vl_keep = _selection(row.Antibody_VL_AA, row.Antibody_VL_Paratope, mask_mode)
        ag_keep = _selection(row.Antigen_AA, row.Antigen_Epitope, mask_mode)
        ab_key = _selected_residues(row.Antibody_VH_AA, vh_keep) + "/" + _selected_residues(row.Antibody_VL_AA, vl_keep)
        ag_key = _selected_residues(row.Antigen_AA, ag_keep)
        if ab_key not in antibody_keys:
            antibody_keys[ab_key] = len(antibody_keys)
        if ag_key not in antigen_keys:
            antigen_keys[ag_key] = len(antigen_keys)
        records.append(
            PairRecord(
                pair_id=str(row.Unique_AG_VH_VL_Hash),
                vh=row.Antibody_VH_AA,
                vl=row.Antibody_VL_AA,
                antigen=row.Antigen_AA,
                vh_keep=vh_keep,
                vl_keep=vl_keep,
                antigen_keep=ag_keep,
                antibody_label=antibody_keys[ab_key],
                antigen_label=antigen_keys[ag_key],
                cluster_id=str(row.Cluster_id),
            )
        )
    return records


class PairDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PairRecord],
        antibody_store: RaggedEmbeddingStore,
        antigen_store: RaggedEmbeddingStore,
    ) -> None:
        self.records = list(records)
        self.antibody_store = antibody_store
        self.antigen_store = antigen_store
        self.antibody_dim = antibody_store.dim
        self.antigen_dim = antigen_store.dim

    def __len__(self) -> int:
        return len(self.records)

    def _antibody_tokens(self, record: PairRecord) -> np.ndarray:
        parts = []
        for sequence, keep in ((record.vh, record.vh_keep), (record.vl, record.vl_keep)):
            if sequence == "" or not keep.any():
                continue
            tokens = self.antibody_store.get(sequence)
            limit = min(tokens.shape[0], keep.size)
            parts.append(tokens[:limit][keep[:limit]])
        if not parts:
            return np.zeros((0, self.antibody_dim), dtype=np.float16)
        return np.concatenate(parts, axis=0)

    def _antigen_tokens(self, record: PairRecord) -> np.ndarray:
        tokens = self.antigen_store.get(record.antigen)
        keep = record.antigen_keep
        limit = min(tokens.shape[0], keep.size)
        selected = tokens[:limit][keep[:limit]]
        if selected.shape[0] == 0:
            return tokens[:limit]
        return selected

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            "antibody": self._antibody_tokens(record),
            "antigen": self._antigen_tokens(record),
            "antibody_label": record.antibody_label,
            "antigen_label": record.antigen_label,
            "pair_id": record.pair_id,
        }


def _stack(blocks: Sequence[np.ndarray], dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = np.array([block.shape[0] for block in blocks], dtype=np.int64)
    if lengths.sum() == 0:
        return torch.zeros((0, dim), dtype=torch.float32), torch.zeros(0, dtype=torch.long)
    flat = np.concatenate([block for block in blocks if block.shape[0] > 0], axis=0)
    index = np.repeat(np.arange(len(blocks), dtype=np.int64), lengths)
    return torch.from_numpy(flat.astype(np.float32)), torch.from_numpy(index)


def collate(batch: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
    antibody_blocks = [item["antibody"] for item in batch]
    antigen_blocks = [item["antigen"] for item in batch]
    antibody_dim = antibody_blocks[0].shape[1]
    antigen_dim = antigen_blocks[0].shape[1]
    ab_tokens, ab_index = _stack(antibody_blocks, antibody_dim)
    ag_tokens, ag_index = _stack(antigen_blocks, antigen_dim)
    return {
        "antibody_tokens": ab_tokens,
        "antibody_index": ab_index,
        "antigen_tokens": ag_tokens,
        "antigen_index": ag_index,
        "antibody_label": torch.tensor([item["antibody_label"] for item in batch], dtype=torch.long),
        "antigen_label": torch.tensor([item["antigen_label"] for item in batch], dtype=torch.long),
        "size": torch.tensor(len(batch), dtype=torch.long),
    }


def make_loader(
    dataset: PairDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    workers = max(0, int(num_workers))
    options: dict[str, object] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=bool(pin_memory),
        generator=generator,
    )
    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 4
    try:
        return DataLoader(dataset, **options)
    except (OSError, RuntimeError, PermissionError):
        options["num_workers"] = 0
        options.pop("persistent_workers", None)
        options.pop("prefetch_factor", None)
        return DataLoader(dataset, **options)
